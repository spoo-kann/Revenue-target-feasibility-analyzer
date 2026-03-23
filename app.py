import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from fpdf import FPDF
import io
import base64
from datetime import datetime

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Revenue Feasibility Analyzer",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# AUTH — file-based user store (no hardcoded credentials)
import hashlib, json, os

USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_pw(pw):
    return hashlib.sha256(pw.strip().encode()).hexdigest()

def do_login(u, p):
    users = load_users()
    return users.get(u.strip()) == hash_pw(p)

def do_register(u, p):
    users = load_users()
    if u.strip() in users:
        return False, "Username already exists."
    if len(u.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(p.strip()) < 4:
        return False, "Password must be at least 4 characters."
    users[u.strip()] = hash_pw(p)
    save_users(users)
    return True, "Account created successfully!"

# Activity log functions
ACTIVITY_FILE = "activity_log.json"

def load_activity():
    if os.path.exists(ACTIVITY_FILE):
        with open(ACTIVITY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_activity(log):
    with open(ACTIVITY_FILE, "w") as f:
        json.dump(log, f)

def log_login(username):
    log = load_activity()
    if username not in log:
        log[username] = {"login_count": 0, "last_login": "", "sessions": []}
    log[username]["login_count"] += 1
    log[username]["last_login"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
    log[username]["sessions"].append(datetime.now().strftime("%d %b %Y, %I:%M %p"))
    log[username]["sessions"] = log[username]["sessions"][-10:]  # keep last 10
    save_activity(log)

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "auth_tab" not in st.session_state:
    st.session_state["auth_tab"] = "login"  ""

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
  --bg:       #060810;
  --surface:  #0d1117;
  --card:     #111827;
  --border:   #1f2937;
  --accent:   #38bdf8;
  --accent2:  #f472b6;
  --accent3:  #34d399;
  --warn:     #fbbf24;
  --danger:   #f87171;
  --text:     #e2e8f0;
  --muted:    #64748b;
  --glow:     rgba(56,189,248,0.15);
}

/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--bg) !important;
  color: var(--text);
}

/* ── Animated grid background ── */
.main > div {
  background-image:
    linear-gradient(rgba(56,189,248,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(56,189,248,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header { visibility: visible; }
.block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1400px; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Animated hero title ── */
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 2.8rem;
  font-weight: 800;
  background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: gradientShift 4s ease infinite alternate;
  background-size: 200% 200%;
  line-height: 1.2;
  margin: 0;
}
@keyframes gradientShift {
  0%   { background-position: 0% 50%; }
  100% { background-position: 100% 50%; }
}

.hero-sub {
  color: var(--muted);
  font-size: 1rem;
  font-weight: 300;
  margin-top: 0.5rem;
  letter-spacing: 0.02em;
}

/* ── Section headers ── */
.section-header {
  font-family: 'Syne', sans-serif;
  font-size: 1.25rem;
  font-weight: 700;
  color: #0a0e1a;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 2rem 0 1rem;
  padding: 0.5rem 1rem 0.5rem;
  background: linear-gradient(90deg, rgba(56,189,248,0.12), transparent);
  border-left: 3px solid var(--accent);
  border-radius: 0 8px 8px 0;
  position: relative;
}

/* ── KPI Cards ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1rem;
  margin: 1rem 0 2rem;
}
.kpi-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.2rem 1.4rem;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  animation: fadeSlideUp 0.5s ease both;
}
.kpi-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 32px var(--glow);
  border-color: var(--accent);
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  opacity: 0;
  transition: opacity 0.2s;
}
.kpi-card:hover::before { opacity: 1; }
.kpi-card::after {
  content: '';
  position: absolute;
  top: -40px; right: -40px;
  width: 80px; height: 80px;
  border-radius: 50%;
  background: var(--glow);
  filter: blur(20px);
}
.kpi-label {
  font-size: 0.7rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--muted);
  margin-bottom: 0.5rem;
}
.kpi-value {
  font-family: 'Syne', sans-serif;
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--accent);
  line-height: 1;
}
.kpi-icon {
  font-size: 1.4rem;
  margin-bottom: 0.4rem;
  display: block;
}

/* ── Metric override ── */
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 1.2rem !important;
  transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s !important;
}
[data-testid="metric-container"]:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 6px 24px var(--glow) !important;
  border-color: var(--accent) !important;
}
[data-testid="stMetricLabel"] > div {
  font-size: 0.72rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 1.55rem !important;
  font-weight: 700 !important;
  color: var(--accent) !important;
}

/* ── Verdict badges ── */
.verdict-achievable {
  background: linear-gradient(135deg, rgba(52,211,153,0.12), rgba(52,211,153,0.04));
  border: 1px solid rgba(52,211,153,0.4);
  border-radius: 14px;
  padding: 1.2rem 1.6rem;
  font-family: 'Syne', sans-serif;
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--accent3);
  display: flex; align-items: center; gap: 0.75rem;
  animation: pulseGreen 2s ease infinite;
}
.verdict-challenging {
  background: linear-gradient(135deg, rgba(251,191,36,0.12), rgba(251,191,36,0.04));
  border: 1px solid rgba(251,191,36,0.4);
  border-radius: 14px;
  padding: 1.2rem 1.6rem;
  font-family: 'Syne', sans-serif;
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--warn);
  display: flex; align-items: center; gap: 0.75rem;
  animation: pulseYellow 2s ease infinite;
}
.verdict-unrealistic {
  background: linear-gradient(135deg, rgba(248,113,113,0.12), rgba(248,113,113,0.04));
  border: 1px solid rgba(248,113,113,0.4);
  border-radius: 14px;
  padding: 1.2rem 1.6rem;
  font-family: 'Syne', sans-serif;
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--danger);
  display: flex; align-items: center; gap: 0.75rem;
  animation: pulseRed 2s ease infinite;
}
@keyframes pulseGreen {
  0%,100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
  50%      { box-shadow: 0 0 16px 2px rgba(52,211,153,0.2); }
}
@keyframes pulseYellow {
  0%,100% { box-shadow: 0 0 0 0 rgba(251,191,36,0); }
  50%      { box-shadow: 0 0 16px 2px rgba(251,191,36,0.2); }
}
@keyframes pulseRed {
  0%,100% { box-shadow: 0 0 0 0 rgba(248,113,113,0); }
  50%      { box-shadow: 0 0 16px 2px rgba(248,113,113,0.2); }
}

/* ── Upload zone — full override, NO blur ── */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div,
[data-testid="stFileDropzone"],
[data-testid="stFileDropzone"] > div {
  filter: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
  opacity: 1 !important;
  background: var(--card) !important;
  color: var(--text) !important;
}
[data-testid="stFileUploader"] section {
  border: 1.5px dashed var(--border) !important;
  border-radius: 14px !important;
  background: var(--card) !important;
  padding: 2rem !important;
}
[data-testid="stFileUploader"] section:hover {
  border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section p,
[data-testid="stFileUploader"] section small {
  color: var(--text) !important;
  opacity: 1 !important;
  filter: none !important;
}
[data-testid="stFileUploader"] section button {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--accent) !important;
  font-weight: 600 !important;
  opacity: 1 !important;
  filter: none !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stFileUploader"] section button:hover {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.88rem;
  font-weight: 500;
  color: var(--muted);
  padding: 0.6rem 1.2rem;
  border-radius: 8px 8px 0 0;
  transition: color 0.2s, background 0.2s;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--accent) !important;
  background: rgba(56,189,248,0.08) !important;
}
[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid var(--border) !important;
  gap: 0.25rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #080c14 !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio label {
  font-size: 0.88rem;
  font-weight: 500;
  padding: 0.5rem 0.8rem;
  border-radius: 8px;
  transition: background 0.15s;
  display: block;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(56,189,248,0.08);
}

/* ── Sidebar logo/brand ── */
.sidebar-brand {
  font-family: 'Syne', sans-serif;
  font-size: 1.1rem;
  font-weight: 800;
  background: linear-gradient(135deg, #38bdf8, #818cf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  padding: 0.5rem 0 0.25rem;
  letter-spacing: -0.01em;
}
.sidebar-version {
  font-size: 0.7rem;
  color: var(--muted);
  margin-bottom: 1rem;
}
.sidebar-divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1rem 0;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div {
  background: var(--accent) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border-radius: 12px !important;
  overflow: hidden;
}

/* ── Info / success / warning / error ── */
[data-testid="stAlert"] { border-radius: 12px !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Number input ── */
[data-testid="stNumberInput"] input {
  background: var(--card) !important;
  border-color: var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif;
}
[data-testid="stNumberInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}

/* ── Fade in animations ── */
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.animate-in {
  animation: fadeSlideUp 0.5s ease both;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Action Plan ── */
.action-plan-wrapper {
  margin-top: 2rem;
  animation: fadeSlideUp 0.6s ease both;
}
.action-plan-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.1rem;
  font-weight: 700;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.action-plan-title.warn  { color: #fbbf24; }
.action-plan-title.danger { color: #f87171; }
.action-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.9rem;
}
.action-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
}
.action-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 24px rgba(56,189,248,0.1);
  border-color: var(--accent);
}
.action-card-top {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}
.action-icon { font-size: 1.3rem; }
.action-label {
  font-family: 'Syne', sans-serif;
  font-size: 0.8rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--accent);
}
.action-text {
  font-size: 0.86rem;
  color: var(--text);
  line-height: 1.55;
}
.action-metric {
  margin-top: 0.6rem;
  font-family: 'Syne', sans-serif;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--accent2);
}
.action-divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.5rem 0 1rem;
}
.revised-target-box {
  background: linear-gradient(135deg, rgba(56,189,248,0.08), rgba(129,140,248,0.05));
  border: 1px solid rgba(56,189,248,0.25);
  border-radius: 14px;
  padding: 1.2rem 1.6rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 1rem;
}
.revised-label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin-bottom: 0.25rem;
}
.revised-value {
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--accent);
}

/* ── Login Page ── */
.login-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 3rem 2.8rem 2.5rem;
  max-width: 420px;
  width: 100%;
  margin: 6rem auto 0;
  animation: fadeSlideUp 0.5s ease both;
  box-shadow: 0 24px 64px rgba(0,0,0,0.4);
}
.login-logo { font-size: 2.8rem; text-align:center; display:block; margin-bottom:0.4rem; }
.login-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.6rem; font-weight: 800; text-align: center;
  background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  margin-bottom: 0.2rem;
}
.login-sub { text-align:center; color:var(--muted); font-size:0.85rem; margin-bottom:2rem; }
.login-label {
  font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--muted); margin-bottom: 0.4rem; display: block;
}
.login-error {
  background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.35);
  border-radius: 10px; padding: 0.75rem 1rem; color: #f87171;
  font-size: 0.85rem; margin-top: 1rem; text-align: center;
}
.login-footer { text-align:center; font-size:0.72rem; color:var(--muted); margin-top:1.5rem; }
.user-badge {
  background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.2);
  border-radius: 10px; padding: 0.6rem 0.9rem; margin-bottom: 0.5rem;
  font-size: 0.82rem; color: var(--text);
}
.user-badge span { color: var(--accent); font-weight: 600; }

/* ── Caption / footer ── */
.app-footer {
  text-align: center;
  font-size: 0.72rem;
  color: var(--muted);
  padding: 2rem 0 0.5rem;
  letter-spacing: 0.05em;
}

/* ── Landing page (no file) ── */
.landing-card {
  background: linear-gradient(135deg, #0d1117 60%, #111827);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 3rem;
  text-align: center;
  max-width: 640px;
  margin: 4rem auto;
  animation: fadeSlideUp 0.6s ease both;
}
.landing-icon { font-size: 3.5rem; margin-bottom: 1rem; display: block; }
.landing-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.8rem;
  font-weight: 800;
  background: linear-gradient(135deg, #38bdf8, #f472b6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.75rem;
}
.landing-desc { color: var(--muted); font-size: 0.95rem; line-height: 1.7; }
.pill {
  display: inline-block;
  background: rgba(56,189,248,0.1);
  border: 1px solid rgba(56,189,248,0.25);
  border-radius: 20px;
  padding: 0.25rem 0.85rem;
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--accent);
  margin: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════
BG      = "#060810"
SURFACE = "#0d1117"
CARD    = "#111827"
BORDER  = "#1f2937"
ACCENT  = "#38bdf8"
ACCENT2 = "#f472b6"
ACCENT3 = "#34d399"
WARN    = "#fbbf24"
DANGER  = "#f87171"
MUTED   = "#64748b"
TEXT    = "#e2e8f0"

# ══════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════
def show_login():
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none!important}"
        ".block-container{max-width:480px!important;padding-top:2rem!important}</style>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='text-align:center;margin-bottom:1.5rem'>"
        "<span style='font-size:2.8rem'>💹</span>"
        "<div style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;"
        "background:linear-gradient(135deg,#38bdf8,#818cf8,#f472b6);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text'>"
        "RevAnalyzer</div>"
        "<div style='color:#64748b;font-size:0.85rem;margin-top:0.2rem'>"
        "Revenue Target Feasibility Analyzer</div>"
        "</div>",
        unsafe_allow_html=True
    )

    # Tab switcher
    tab_col1, tab_col2 = st.columns(2)
    with tab_col1:
        if st.button("🔑  Sign In", use_container_width=True,
                     type="primary" if st.session_state["auth_tab"] == "login" else "secondary"):
            st.session_state["auth_tab"] = "login"
            st.rerun()
    with tab_col2:
        if st.button("📝  Create Account", use_container_width=True,
                     type="primary" if st.session_state["auth_tab"] == "register" else "secondary"):
            st.session_state["auth_tab"] = "register"
            st.rerun()

    st.markdown("<div style='margin:0.8rem 0'></div>", unsafe_allow_html=True)

    if st.session_state["auth_tab"] == "login":
        # ── LOGIN FORM ──
        with st.form("login_form", clear_on_submit=False):
            st.markdown("<span class=\"login-label\">Username</span>", unsafe_allow_html=True)
            username = st.text_input("lu", placeholder="Enter your username", label_visibility="collapsed")
            st.markdown("<span class=\"login-label\" style=\"margin-top:0.8rem;display:block\">Password</span>", unsafe_allow_html=True)
            password = st.text_input("lp", placeholder="Enter your password", type="password", label_visibility="collapsed")
            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Sign In  →", use_container_width=True)
            if submitted:
                if not username.strip() or not password.strip():
                    st.markdown("<div class=\"login-error\">❌ &nbsp; Please enter username and password.</div>", unsafe_allow_html=True)
                elif do_login(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"]  = username.strip()
                    log_login(username.strip())
                    st.rerun()
                else:
                    users = load_users()
                    if username.strip() not in users:
                        st.markdown("<div class=\"login-error\">❌ &nbsp; Username not found. Please create an account.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class=\"login-error\">❌ &nbsp; Incorrect password. Try again.</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;font-size:0.8rem;color:#475569;margin-top:0.8rem'>"
            "Don't have an account? Click <strong style='color:#38bdf8'>Create Account</strong> above.</div>",
            unsafe_allow_html=True
        )

    else:
        # ── REGISTER FORM ──
        with st.form("register_form", clear_on_submit=True):
            st.markdown("<span class=\"login-label\">Choose a Username</span>", unsafe_allow_html=True)
            new_user = st.text_input("ru", placeholder="Min. 3 characters", label_visibility="collapsed")
            st.markdown("<span class=\"login-label\" style=\"margin-top:0.8rem;display:block\">Choose a Password</span>", unsafe_allow_html=True)
            new_pass = st.text_input("rp", placeholder="Min. 4 characters", type="password", label_visibility="collapsed")
            st.markdown("<span class=\"login-label\" style=\"margin-top:0.8rem;display:block\">Confirm Password</span>", unsafe_allow_html=True)
            confirm_pass = st.text_input("cp", placeholder="Re-enter password", type="password", label_visibility="collapsed")
            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            reg_submitted = st.form_submit_button("Create Account  →", use_container_width=True)
            if reg_submitted:
                if not new_user.strip() or not new_pass.strip():
                    st.markdown("<div class=\"login-error\">❌ &nbsp; Please fill all fields.</div>", unsafe_allow_html=True)
                elif new_pass != confirm_pass:
                    st.markdown("<div class=\"login-error\">❌ &nbsp; Passwords do not match.</div>", unsafe_allow_html=True)
                else:
                    success, msg = do_register(new_user, new_pass)
                    if success:
                        st.success(f"✅ {msg} Please sign in.")
                        st.session_state["auth_tab"] = "login"
                        st.rerun()
                    else:
                        st.markdown(f"<div class=\"login-error\">❌ &nbsp; {msg}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;font-size:0.8rem;color:#475569;margin-top:0.8rem'>"
            "Already have an account? Click <strong style='color:#38bdf8'>Sign In</strong> above.</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class=\"login-footer\">Built with Streamlit &nbsp;·&nbsp; Data processed locally &nbsp;·&nbsp; Private</div>",
        unsafe_allow_html=True
    )

if not st.session_state["logged_in"]:
    show_login()
    st.stop()

def style_fig(fig, axes=None):
    fig.patch.set_facecolor(SURFACE)
    fig.patch.set_alpha(0.0)
    axes = axes or [fig.gca()]
    for ax in axes:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=BORDER, linewidth=0.6, alpha=0.6)
        ax.xaxis.grid(False)
    fig.tight_layout(pad=1.5)
    return fig

def kpi_html(icon, label, value, delay=0):
    return f"""
    <div class="kpi-card" style="animation-delay:{delay}s">
      <span class="kpi-icon">{icon}</span>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
    </div>"""

def section(icon, title):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SIDEBAR — Settings & Confidence Band removed
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-brand">💹 RevAnalyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-version">v3.0 · Linear Regression</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("**📂 Data**")
    uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"], label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown("**🗂️ Pages**")
    page = st.radio("", ["🏠  Dashboard", "📊  Sales Analytics", "📥  Download Report", "👤  Activity Log"], label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    uname = st.session_state.get("username", "")
    st.markdown(f'<div class="user-badge">👤 Signed in as <span>{uname}</span></div>', unsafe_allow_html=True)
    if st.button("🚪  Sign Out", use_container_width=True):
        for k in ["logged_in","username","target","forecast_period","predicted_total"]:
            st.session_state.pop(k, None)
        st.session_state["logged_in"] = False
        st.rerun()
    st.markdown('<div style="font-size:0.72rem;color:#475569;line-height:1.6;margin-top:0.5rem">Built with Streamlit<br>Data processed locally · Private</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LANDING — no file uploaded
# ══════════════════════════════════════════════════════════
if not uploaded_file:
    st.markdown("""
    <div class="landing-card">
      <span class="landing-icon">💹</span>
      <div class="landing-title">Revenue Target Feasibility Analyzer</div>
      <p class="landing-desc">
        Upload your sales CSV to explore historical trends, build a Linear Regression forecast,
        and instantly evaluate whether your revenue target is achievable.
      </p>
      <div style="margin-top:1.5rem">
        <span class="pill">📈 Forecasting</span>
        <span class="pill">🎯 Feasibility Score</span>
        <span class="pill">🔎 EDA</span>
        <span class="pill">📊 KPIs</span>
      </div>
      <p style="margin-top:1.5rem;font-size:0.82rem;color:#475569">
        ← Upload a CSV from the sidebar to begin<br>
        <span style="font-size:0.75rem">Expected: Order Date · Sales · Profit · Order ID · Category · Region</span>
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding='latin1')

    # ── Auto Column Mapper ─────────────────────────
    # Normalize all column names to strip spaces and lower for matching
    col_map = {c.lower().strip().replace(" ", "").replace("_", ""): c for c in df.columns}

    def find_col(candidates):
        for c in candidates:
            key = c.lower().replace(" ", "").replace("_", "")
            if key in col_map:
                return col_map[key]
        return None

    # Map to standard names
    rename = {}
    date_col   = find_col(["Order Date", "OrderDate", "Date", "order_date", "InvoiceDate", "date", "Trans Date"])
    sales_col  = find_col(["Sales", "Revenue", "Amount", "Total", "Sale Amount", "GrossRevenue", "amount", "sales"])
    profit_col = find_col(["Profit", "Margin", "Net Profit", "profit", "NetProfit", "GrossProfit"])
    orderid    = find_col(["Order ID", "OrderID", "Order_ID", "InvoiceNo", "Transaction ID", "order_id", "id"])
    cat_col    = find_col(["Category", "category", "Product Category", "Type", "Segment", "Department"])
    region_col = find_col(["Region", "region", "State", "City", "Location", "Geography", "Country"])

    if date_col   and date_col   != "Order Date":   rename[date_col]   = "Order Date"
    if sales_col  and sales_col  != "Sales":         rename[sales_col]  = "Sales"
    if profit_col and profit_col != "Profit":        rename[profit_col] = "Profit"
    if orderid    and orderid    != "Order ID":      rename[orderid]    = "Order ID"
    if cat_col    and cat_col    != "Category":      rename[cat_col]    = "Category"
    if region_col and region_col != "Region":        rename[region_col] = "Region"

    if rename:
        df = df.rename(columns=rename)

    # Add missing required columns with defaults
    if "Order ID" not in df.columns:
        df["Order ID"] = range(1, len(df) + 1)
    if "Profit" not in df.columns:
        df["Profit"] = df["Sales"] * 0.15 if "Sales" in df.columns else 0
    if "Category" not in df.columns:
        df["Category"] = "General"
    if "Region" not in df.columns:
        df["Region"] = "Unknown"

    # ── Clean ─────────────────────────────────────
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Sales']      = pd.to_numeric(df['Sales'], errors='coerce')
    df['Profit']     = pd.to_numeric(df['Profit'], errors='coerce')
    df = df.drop_duplicates()
    df = df.dropna(subset=['Order Date', 'Sales'])
    df['Profit']     = df['Profit'].fillna(df['Sales'] * 0.15)
    df['Month']      = df['Order Date'].dt.to_period('M')
    df['Year']       = df['Order Date'].dt.year
    return df

df = load_data(uploaded_file)

monthly_revenue = (
    df.groupby("Month")["Sales"].sum()
    .reset_index().sort_values("Month")
)
monthly_revenue["Month_str"]  = monthly_revenue["Month"].astype(str)
monthly_revenue["Time_Index"] = np.arange(len(monthly_revenue))

# ══════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════
if page == "🏠  Dashboard":

    # Hero
    st.markdown("""
    <div class="animate-in" style="padding:1.5rem 0 0.5rem">
      <div class="hero-title">Revenue Target<br>Feasibility Analyzer</div>
      <p class="hero-sub">Forecast · Analyze · Decide with confidence</p>
    </div>
    """, unsafe_allow_html=True)

    # Data preview
    with st.expander("🗂️ Dataset Preview & Cleaning Report", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows",      f"{len(df):,}")
        c2.metric("Unique Orders",   f"{df['Order ID'].nunique():,}")
        c3.metric("Date Range",      f"{df['Order Date'].min().strftime('%b %Y')} → {df['Order Date'].max().strftime('%b %Y')}")
        st.dataframe(df.head(8), use_container_width=True)

    # ── Data Quality Score ────────────────────────
    total_rows      = len(df)
    missing_pct     = df.isnull().mean().mean() * 100
    dup_pct         = (df.duplicated().sum() / max(total_rows, 1)) * 100
    months_covered  = monthly_revenue["Month"].nunique()
    date_gaps       = months_covered / max((df['Order Date'].max() - df['Order Date'].min()).days / 30, 1) * 100
    date_gap_score  = min(date_gaps, 100)

    missing_score   = max(0, 100 - missing_pct * 5)
    dup_score       = max(0, 100 - dup_pct * 5)
    size_score      = min(100, (total_rows / 1000) * 50 + 50)
    quality_score   = int((missing_score * 0.35) + (dup_score * 0.25) + (size_score * 0.2) + (date_gap_score * 0.2))

    q_color  = "#34d399" if quality_score >= 80 else ("#fbbf24" if quality_score >= 55 else "#f87171")
    q_label  = "Excellent" if quality_score >= 80 else ("Good" if quality_score >= 55 else "Needs Improvement")
    q_bar    = quality_score

    section("🔬", "Data Quality Score")
    st.markdown(
        f"<div style='background:var(--card);border:1px solid var(--border);border-radius:16px;"
        f"padding:1.4rem 1.8rem;margin-bottom:1.5rem'>"
        f"<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;margin-bottom:1rem'>"
        f"<div>"
        f"<div style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);margin-bottom:0.3rem'>Dataset Health</div>"
        f"<div style='font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;color:{q_color}'>{quality_score}%</div>"
        f"<div style='font-size:0.85rem;color:{q_color};font-weight:600'>{q_label}</div>"
        f"</div>"
        f"<div style='display:flex;gap:1.5rem;flex-wrap:wrap'>"
        f"<div style='text-align:center'><div style='font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em'>Missing Values</div>"
        f"<div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:{'#34d399' if missing_pct < 5 else '#f87171'}'>{missing_pct:.1f}%</div></div>"
        f"<div style='text-align:center'><div style='font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em'>Duplicates</div>"
        f"<div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:{'#34d399' if dup_pct < 2 else '#f87171'}'>{dup_pct:.1f}%</div></div>"
        f"<div style='text-align:center'><div style='font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em'>Total Rows</div>"
        f"<div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:var(--accent)'>{total_rows:,}</div></div>"
        f"<div style='text-align:center'><div style='font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em'>Months Covered</div>"
        f"<div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:var(--accent)'>{months_covered}</div></div>"
        f"</div></div>"
        f"<div style='background:var(--border);border-radius:999px;height:8px;overflow:hidden'>"
        f"<div style='background:{q_color};width:{q_bar}%;height:100%;border-radius:999px;"
        f"transition:width 0.8s ease'></div></div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # ── KPIs ─────────────────────────────────────
    total_sales   = df["Sales"].sum()
    total_profit  = df["Profit"].sum()
    total_orders  = df["Order ID"].nunique()
    profit_margin = (total_profit / total_sales * 100) if total_sales else 0
    avg_order_val = total_sales / total_orders if total_orders else 0

    section("📌", "Business KPIs")
    st.markdown(f"""
    <div class="kpi-grid">
      {kpi_html("💰", "Total Sales",     f"${total_sales/1e6:.2f}M",   0.0)}
      {kpi_html("📈", "Total Profit",    f"${total_profit/1e3:.0f}K",  0.08)}
      {kpi_html("🛒", "Total Orders",    f"{total_orders:,}",           0.16)}
      {kpi_html("📊", "Profit Margin",   f"{profit_margin:.1f}%",       0.24)}
      {kpi_html("🎫", "Avg Order Value", f"${avg_order_val:,.0f}",      0.32)}
    </div>
    """, unsafe_allow_html=True)

    # ── Forecasting ───────────────────────────────
    section("🤖", "Revenue Forecasting — ML Pipeline")

    col_sl, _ = st.columns([1, 2])
    with col_sl:
        forecast_period = st.slider("Forecast Months Ahead", 1, 24, 6)

    X = monthly_revenue[["Time_Index"]]
    y = monthly_revenue["Sales"]

    # ── Scikit-learn Pipeline ──────────────────────
    revenue_pipeline = Pipeline([
        ("poly",      PolynomialFeatures(degree=1, include_bias=False)),
        ("scaler",    StandardScaler()),
        ("regressor", LinearRegression())
    ])
    revenue_pipeline.fit(X, y)
    y_pred_hist  = revenue_pipeline.predict(X)
    r2           = r2_score(y, y_pred_hist)
    mae          = mean_absolute_error(y, y_pred_hist)
    future_index = np.arange(len(monthly_revenue), len(monthly_revenue) + forecast_period).reshape(-1,1)
    predictions  = np.clip(revenue_pipeline.predict(future_index), 0, None)
    residuals    = y.values - y_pred_hist
    std_res      = np.std(residuals)
    z            = 1.96
    lower        = predictions - z * std_res
    upper        = predictions + z * std_res
    predicted_total = float(predictions.sum())
    st.session_state['forecast_period'] = forecast_period
    st.session_state['predicted_total'] = predicted_total

    # Cross-validation score
    cv_scores = cross_val_score(revenue_pipeline, X, y, cv=min(3, len(X)), scoring='r2')
    cv_mean   = cv_scores.mean()
    cv_color  = "#34d399" if cv_mean > 0.7 else ("#fbbf24" if cv_mean > 0.4 else "#f87171")
    cv_label  = "Good fit" if cv_mean > 0.7 else ("Moderate fit" if cv_mean > 0.4 else "Weak fit")

    # ── Pipeline Flowchart ────────────────────────
    st.markdown(
        "<div style='margin:1rem 0 1.2rem;overflow-x:auto'>"
        "<div style='display:flex;align-items:center;gap:0;min-width:max-content;padding:0.2rem 0'>"
        "<div style='background:#0d1117;border:1px solid #38bdf8;border-radius:10px;padding:0.55rem 1rem;font-size:0.78rem;color:#38bdf8;font-weight:600;font-family:Syne,sans-serif;white-space:nowrap'>📂 Load CSV</div>"
        "<div style='color:#334155;font-size:1rem;margin:0 0.3rem'>──▶</div>"
        "<div style='background:#0d1117;border:1px solid #818cf8;border-radius:10px;padding:0.55rem 1rem;font-size:0.78rem;color:#818cf8;font-weight:600;font-family:Syne,sans-serif;white-space:nowrap'>🧹 Clean Data</div>"
        "<div style='color:#334155;font-size:1rem;margin:0 0.3rem'>──▶</div>"
        "<div style='background:#0d1117;border:1px solid #f472b6;border-radius:10px;padding:0.55rem 1rem;font-size:0.78rem;color:#f472b6;font-weight:600;font-family:Syne,sans-serif;white-space:nowrap'>🔢 PolynomialFeatures</div>"
        "<div style='color:#334155;font-size:1rem;margin:0 0.3rem'>──▶</div>"
        "<div style='background:#0d1117;border:1px solid #fbbf24;border-radius:10px;padding:0.55rem 1rem;font-size:0.78rem;color:#fbbf24;font-weight:600;font-family:Syne,sans-serif;white-space:nowrap'>📐 StandardScaler</div>"
        "<div style='color:#334155;font-size:1rem;margin:0 0.3rem'>──▶</div>"
        "<div style='background:#0d1117;border:1px solid #38bdf8;border-radius:10px;padding:0.55rem 1rem;font-size:0.78rem;color:#38bdf8;font-weight:600;font-family:Syne,sans-serif;white-space:nowrap'>🤖 LinearRegression</div>"
        "<div style='color:#334155;font-size:1rem;margin:0 0.3rem'>──▶</div>"
        "<div style='background:#0d1117;border:1px solid #34d399;border-radius:10px;padding:0.55rem 1rem;font-size:0.78rem;color:#34d399;font-weight:600;font-family:Syne,sans-serif;white-space:nowrap'>🎯 Feasibility Verdict</div>"
        "</div></div>",
        unsafe_allow_html=True
    )

    # ── Pipeline pills ────────────────────────────
    st.markdown(
        f"<div style='display:flex;gap:0.75rem;margin-bottom:1.2rem;flex-wrap:wrap;align-items:center'>"
        f"<span class='pill'>🧠 Sklearn Pipeline</span>"
        f"<span class='pill'>📐 StandardScaler</span>"
        f"<span class='pill'>🔢 PolynomialFeatures</span>"
        f"<span class='pill'>Forecast = {forecast_period} months</span>"
        f"<span class='pill' style='color:{cv_color};border-color:{cv_color}33;background:{cv_color}11'>"
        f"CV R² = {cv_mean:.3f} — {cv_label}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Forecast chart — Interactive Plotly with hover tooltips
    hist_x      = monthly_revenue["Time_Index"].values
    hist_labels = monthly_revenue["Month_str"].values
    fut_x       = np.arange(len(monthly_revenue), len(monthly_revenue) + forecast_period)
    fut_labels  = [f"M+{i+1}" for i in range(forecast_period)]

    fig_plotly = go.Figure()

    # Confidence band (fill between lower/upper)
    fig_plotly.add_trace(go.Scatter(
        x=list(fut_labels) + list(fut_labels[::-1]),
        y=list(np.clip(upper, 0, None)) + list(np.clip(lower, 0, None)[::-1]),
        fill='toself',
        fillcolor='rgba(244,114,182,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        showlegend=False,
        name='Confidence Band'
    ))

    # Historical revenue line
    fig_plotly.add_trace(go.Scatter(
        x=list(hist_labels),
        y=list(y.values),
        mode='lines+markers',
        name='Historical Revenue',
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=5, color=ACCENT),
        hovertemplate='<b>%{x}</b><br>Revenue: <b>$%{y:,.0f}</b><extra></extra>'
    ))

    # Trend line
    fig_plotly.add_trace(go.Scatter(
        x=list(hist_labels),
        y=list(y_pred_hist),
        mode='lines',
        name='Trend Line',
        line=dict(color='#94a3b8', width=1.2, dash='dot'),
        hovertemplate='<b>%{x}</b><br>Trend: <b>$%{y:,.0f}</b><extra></extra>'
    ))

    # Forecast line
    fig_plotly.add_trace(go.Scatter(
        x=list(fut_labels),
        y=list(predictions),
        mode='lines+markers',
        name='Forecast',
        line=dict(color=ACCENT2, width=2.5),
        marker=dict(size=7, color=ACCENT2, symbol='circle'),
        hovertemplate='<b>%{x}</b><br>Forecast: <b>$%{y:,.0f}</b><extra></extra>'
    ))

    # Vertical divider line annotation (using shape — works with string x-axis)
    fig_plotly.add_shape(
        type="line",
        x0=hist_labels[-1], x1=hist_labels[-1],
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=BORDER, width=1.5, dash="dash")
    )
    fig_plotly.add_annotation(
        x=hist_labels[-1],
        y=1.0,
        xref="x", yref="paper",
        text="▶ Forecast",
        showarrow=False,
        font=dict(color=ACCENT2, size=11, family="DM Sans"),
        xanchor="left",
        yanchor="top",
        xshift=8
    )

    fig_plotly.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CARD,
        font=dict(family='DM Sans', color=TEXT, size=11),
        height=340,
        margin=dict(l=60, r=30, t=30, b=50),
        legend=dict(
            bgcolor=CARD,
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(size=11, color=TEXT),
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='left', x=0
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e293b',
            bordercolor=ACCENT,
            font=dict(family='DM Sans', size=12, color=TEXT)
        ),
        xaxis=dict(
            showgrid=False,
            color=MUTED,
            tickfont=dict(size=10, color=MUTED),
            linecolor=BORDER,
            tickangle=-35
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=BORDER,
            gridwidth=0.6,
            color=MUTED,
            tickfont=dict(size=10, color=MUTED),
            linecolor=BORDER,
            tickprefix='$',
            tickformat=',.0f'
        )
    )

    st.plotly_chart(fig_plotly, use_container_width=True)

    # ── Target Feasibility ────────────────────────
    section("🎯", "Target Feasibility Analysis")

    col_inp, _ = st.columns([1, 1])
    with col_inp:
        target = st.number_input(
            f"Revenue Target — next {forecast_period} month(s)",
            min_value=0.0, step=10000.0, format="%.2f",
            help="Enter your desired revenue goal for the forecast window."
        )

    if target > 0:
        st.session_state['target'] = target
        feasibility_score = (predicted_total / target) * 100
        gap     = predicted_total - target
        gap_pct = (gap / target) * 100

        fa, fb, fc, fd = st.columns(4)
        fa.metric("Predicted Revenue", f"${predicted_total:,.0f}")
        fb.metric("Target Revenue",    f"${target:,.0f}")
        fc.metric("Feasibility Score", f"{feasibility_score:.1f}%")
        fd.metric("Gap", f"${gap:+,.0f}", delta=f"{gap_pct:+.1f}%",
                  delta_color="normal" if gap >= 0 else "inverse")

        st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

        if feasibility_score >= 100:
            st.markdown(f'<div class="verdict-achievable">✅ &nbsp; Target is <strong>Achievable</strong> — Forecast exceeds target by {abs(gap_pct):.1f}%</div>', unsafe_allow_html=True)
        elif feasibility_score >= 80:
            st.markdown(f'<div class="verdict-challenging">⚠️ &nbsp; Target is <strong>Challenging</strong> — You\'re {abs(gap_pct):.1f}% short; a strong push is needed.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-unrealistic">❌ &nbsp; Target is <strong>Unrealistic</strong> — Forecast falls {abs(gap_pct):.1f}% below target based on current trends.</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin:1.2rem 0'></div>", unsafe_allow_html=True)

        # Gauge chart
        fig_g, ax_g = plt.subplots(figsize=(9, 1.8))
        fig_g.patch.set_facecolor("none")
        ax_g.set_facecolor(CARD)

        score_capped = min(feasibility_score, 150)
        bar_clr = ACCENT3 if feasibility_score >= 100 else (WARN if feasibility_score >= 80 else DANGER)

        ax_g.barh([""], [150], color=BORDER, height=0.5, zorder=1)
        ax_g.barh([""], [80],  color=(0.973, 0.443, 0.443, 0.15), height=0.5, zorder=2)
        ax_g.barh([""], [100], color=(0.984, 0.749, 0.141, 0.12), height=0.5, left=80,  zorder=2)
        ax_g.barh([""], [50],  color=(0.204, 0.831, 0.600, 0.12), height=0.5, left=100, zorder=2)
        ax_g.barh([""], [score_capped], color=bar_clr, height=0.38, zorder=3,
                  linewidth=0, alpha=0.92)

        ax_g.axvline(x=100, color="#ffffff", lw=1.5, ls='--', alpha=0.5, zorder=4)

        ax_g.text(40, 0, "Unrealistic",  color=DANGER,  va='center', ha='center', fontsize=7.5, alpha=0.7)
        ax_g.text(90, 0, "Challenging",  color=WARN,    va='center', ha='center', fontsize=7.5, alpha=0.7)
        ax_g.text(125,0, "Achievable",   color=ACCENT3, va='center', ha='center', fontsize=7.5, alpha=0.7)

        score_x = max(score_capped - 3, 8)
        ax_g.text(score_x, 0, f"{feasibility_score:.1f}%",
                  color="white", va='center', ha='right',
                  fontweight='bold', fontsize=11, zorder=5)

        ax_g.set_xlim(0, 150)
        ax_g.set_xlabel("Feasibility Score (%)", color=MUTED, fontsize=8.5)
        for sp in ax_g.spines.values():
            sp.set_edgecolor(BORDER)
        ax_g.tick_params(colors=MUTED, labelsize=8)
        ax_g.yaxis.set_visible(False)
        ax_g.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
        ax_g.yaxis.grid(False)
        fig_g.tight_layout(pad=1.2)
        st.pyplot(fig_g, use_container_width=True)

        # ── What-If Scenario Simulator ────────────────
        st.markdown("<div style='margin:2rem 0 0'></div>", unsafe_allow_html=True)
        section("🔮", "What-If Scenario Simulator")
        st.markdown(
            "<div style='font-size:0.88rem;color:var(--muted);margin-bottom:1rem'>"
            "Adjust growth assumptions below to see how they change your forecast and feasibility verdict in real time."
            "</div>",
            unsafe_allow_html=True
        )

        wa, wb = st.columns(2)
        with wa:
            growth_rate = st.slider(
                "Monthly Sales Growth (%)",
                min_value=-20, max_value=50, value=0, step=1,
                help="Simulate what happens if sales grow or decline each month"
            )
        with wb:
            cost_change = st.slider(
                "Cost Reduction (%)",
                min_value=-20, max_value=30, value=0, step=1,
                help="Simulate the effect of reducing costs (improves profit margin)"
            )

        # Apply growth to predictions
        growth_multipliers  = np.array([(1 + growth_rate / 100) ** (i + 1) for i in range(forecast_period)])
        adjusted_preds      = predictions * growth_multipliers
        adjusted_total      = float(adjusted_preds.sum())
        adjusted_margin     = profit_margin * (1 + cost_change / 100)
        adjusted_profit     = total_profit * (1 + cost_change / 100)

        # Recalculate feasibility
        adj_score   = (adjusted_total / target * 100)
        adj_gap     = adjusted_total - target
        adj_gap_pct = (adj_gap / target * 100)

        if adj_score >= 100:
            adj_verdict = "ACHIEVABLE"
            adj_vcolor  = "#34d399"
            adj_icon    = "✅"
        elif adj_score >= 80:
            adj_verdict = "CHALLENGING"
            adj_vcolor  = "#fbbf24"
            adj_icon    = "⚠️"
        else:
            adj_verdict = "UNREALISTIC"
            adj_vcolor  = "#f87171"
            adj_icon    = "❌"

        # Show comparison
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("Adjusted Forecast",  f"${adjusted_total:,.0f}",
                   delta=f"${adjusted_total - predicted_total:+,.0f} vs base")
        wc2.metric("Adjusted Margin",    f"{adjusted_margin:.1f}%",
                   delta=f"{adjusted_margin - profit_margin:+.1f}%")
        wc3.metric("Feasibility Score",  f"{adj_score:.1f}%",
                   delta=f"{adj_score - feasibility_score:+.1f}% vs base")
        wc4.metric("Gap to Target",      f"${adj_gap:+,.0f}",
                   delta=f"{adj_gap_pct:+.1f}%",
                   delta_color="normal" if adj_gap >= 0 else "inverse")

        st.markdown("<div style='margin:0.8rem 0'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:{'rgba(52,211,153,0.08)' if adj_verdict=='ACHIEVABLE' else ('rgba(251,191,36,0.08)' if adj_verdict=='CHALLENGING' else 'rgba(248,113,113,0.08)')};"
            f"border:1px solid {adj_vcolor}55;border-radius:12px;padding:1rem 1.4rem;"
            f"display:flex;align-items:center;gap:0.75rem'>"
            f"<span style='font-size:1.3rem'>{adj_icon}</span>"
            f"<div>"
            f"<div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:{adj_vcolor}'>"
            f"With {growth_rate:+d}% monthly growth & {cost_change:+d}% cost change — Target is {adj_verdict}"
            f"</div>"
            f"<div style='font-size:0.82rem;color:var(--muted);margin-top:0.2rem'>"
            f"Adjusted forecast: ${adjusted_total:,.0f} vs target: ${target:,.0f} &nbsp;|&nbsp; Gap: ${adj_gap:+,.0f}"
            f"</div>"
            f"</div></div>",
            unsafe_allow_html=True
        )

        # ── Target Tracking Timeline ──────────────────
        st.markdown("<div style='margin:2rem 0 0'></div>", unsafe_allow_html=True)
        section("📅", "Target Tracking Timeline")
        st.markdown(
            "<div style='font-size:0.88rem;color:var(--muted);margin-bottom:1rem'>"
            "Month-by-month forecast vs your target — see exactly when you are on track and when you fall short."
            "</div>", unsafe_allow_html=True
        )

        target_line   = [target / forecast_period] * forecast_period
        cumulative_fc = np.cumsum(predictions)
        cumulative_tg = np.cumsum(target_line)
        hit_month     = next((i+1 for i, v in enumerate(cumulative_fc) if v >= target), None)

        fig_tt = go.Figure()
        fig_tt.add_trace(go.Scatter(
            x=fut_labels, y=list(cumulative_tg),
            mode='lines', name='Cumulative Target',
            line=dict(color=WARN, width=2, dash='dash'),
            hovertemplate='<b>%{x}</b><br>Target: <b>$%{y:,.0f}</b><extra></extra>'
        ))
        fig_tt.add_trace(go.Scatter(
            x=fut_labels, y=list(cumulative_fc),
            mode='lines+markers', name='Cumulative Forecast',
            line=dict(color=ACCENT, width=2.5),
            marker=dict(
                size=8, color=[ACCENT3 if v >= t else DANGER
                               for v, t in zip(cumulative_fc, cumulative_tg)],
                symbol='circle'
            ),
            hovertemplate='<b>%{x}</b><br>Forecast: <b>$%{y:,.0f}</b><extra></extra>'
        ))
        if hit_month:
            fig_tt.add_vline(
                x=fut_labels[hit_month-1],
                line_width=1.5, line_dash="dot", line_color=ACCENT3,
                annotation_text=f"Target Hit at M+{hit_month}",
                annotation_font_color=ACCENT3, annotation_font_size=10
            )
        fig_tt.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=CARD,
            font=dict(family='DM Sans', color=TEXT, size=11),
            height=280, margin=dict(l=60, r=30, t=30, b=40),
            legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                        font=dict(size=11, color=TEXT), orientation='h',
                        yanchor='bottom', y=1.02, xanchor='left', x=0),
            hovermode='x unified',
            hoverlabel=dict(bgcolor='#1e293b', bordercolor=ACCENT,
                            font=dict(family='DM Sans', size=12, color=TEXT)),
            xaxis=dict(showgrid=False, color=MUTED, tickfont=dict(size=10, color=MUTED), linecolor=BORDER),
            yaxis=dict(showgrid=True, gridcolor=BORDER, gridwidth=0.6, color=MUTED,
                       tickfont=dict(size=10, color=MUTED), linecolor=BORDER,
                       tickprefix='$', tickformat=',.0f')
        )
        st.plotly_chart(fig_tt, use_container_width=True)

        if hit_month:
            st.markdown(
                f"<div style='background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.3);"
                f"border-radius:10px;padding:0.8rem 1.2rem;font-size:0.88rem;color:#34d399'>"
                f"✅ Based on forecast, cumulative revenue will hit <strong>${target:,.0f}</strong> "
                f"by <strong>Month +{hit_month}</strong> ({fut_labels[hit_month-1]})</div>",
                unsafe_allow_html=True
            )
        else:
            shortfall = target - float(cumulative_fc[-1])
            st.markdown(
                f"<div style='background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.3);"
                f"border-radius:10px;padding:0.8rem 1.2rem;font-size:0.88rem;color:#f87171'>"
                f"❌ Cumulative forecast does not reach target within {forecast_period} months. "
                f"Shortfall: <strong>${shortfall:,.0f}</strong></div>",
                unsafe_allow_html=True
            )

        # ── Risk Factor Generator ─────────────────────
        st.markdown("<div style='margin:2rem 0 0'></div>", unsafe_allow_html=True)
        section("⚠️", "Risk Factor Analysis")

        risks = []

        # Risk 1: Revenue concentration
        top_cat_s     = df.groupby("Category")["Sales"].sum()
        top_cat_pct   = (top_cat_s.max() / top_cat_s.sum() * 100)
        top_cat_name  = top_cat_s.idxmax()
        if top_cat_pct > 55:
            risks.append(("HIGH", "Revenue Concentration",
                f"{top_cat_name} drives {top_cat_pct:.1f}% of total revenue — over-dependence on a single category creates vulnerability if demand shifts."))
        elif top_cat_pct > 40:
            risks.append(("MEDIUM", "Revenue Concentration",
                f"{top_cat_name} drives {top_cat_pct:.1f}% of revenue — moderate concentration. Consider diversifying category mix."))

        # Risk 2: Profit margin
        overall_margin = (df["Profit"].sum() / df["Sales"].sum() * 100) if df["Sales"].sum() else 0
        if overall_margin < 10:
            risks.append(("HIGH", "Low Profit Margin",
                f"Profit margin is {overall_margin:.1f}% — critically low. Any cost increase or discount strategy could push margins negative."))
        elif overall_margin < 15:
            risks.append(("MEDIUM", "Profit Margin Warning",
                f"Profit margin is {overall_margin:.1f}% — below healthy threshold of 15%. Review pricing and cost structure."))

        # Risk 3: Revenue decline trend
        if len(monthly_revenue) >= 3:
            last3 = monthly_revenue["Sales"].tail(3).values
            if last3[2] < last3[1] < last3[0]:
                risks.append(("HIGH", "3-Month Declining Trend",
                    f"Revenue has declined 3 consecutive months: ${last3[0]:,.0f} → ${last3[1]:,.0f} → ${last3[2]:,.0f}. Forecast reliability is reduced."))
            elif last3[2] < last3[0]:
                risks.append(("MEDIUM", "Short-term Revenue Dip",
                    f"Revenue in the most recent month (${last3[2]:,.0f}) is below 2 months ago (${last3[0]:,.0f}). Monitor closely."))

        # Risk 4: Feasibility gap
        if feasibility_score < 60:
            risks.append(("HIGH", "Unrealistic Target",
                f"Feasibility score of {feasibility_score:.1f}% indicates target is significantly above forecast. Risk of team burnout chasing an unachievable number."))
        elif feasibility_score < 80:
            risks.append(("MEDIUM", "Challenging Target",
                f"Feasibility score of {feasibility_score:.1f}% means target requires above-trend performance. Achievable but requires focused effort."))

        # Risk 5: Data quality
        missing_pct_r = df.isnull().mean().mean() * 100
        if missing_pct_r > 10:
            risks.append(("MEDIUM", "Data Quality Risk",
                f"{missing_pct_r:.1f}% of data has missing values. Forecast accuracy may be affected. Clean the dataset for better predictions."))

        # Risk 6: Revenue volatility
        rev_std  = monthly_revenue["Sales"].std()
        rev_mean = monthly_revenue["Sales"].mean()
        cv       = (rev_std / rev_mean * 100) if rev_mean else 0
        if cv > 40:
            risks.append(("HIGH", "High Revenue Volatility",
                f"Revenue swings ±{rev_std:,.0f} monthly (CV={cv:.0f}%). High volatility makes forecasting unreliable — widen your planning range."))
        elif cv > 20:
            risks.append(("MEDIUM", "Moderate Volatility",
                f"Revenue varies ±{rev_std:,.0f} monthly (CV={cv:.0f}%). Factor this uncertainty into target setting."))

        if not risks:
            st.markdown(
                "<div style='background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.25);"
                "border-radius:12px;padding:1rem 1.4rem;color:#34d399;font-size:0.9rem'>"
                "✅ No significant risk factors detected. Your data quality, margin, and trend are all healthy.</div>",
                unsafe_allow_html=True
            )
        else:
            risk_html = ""
            for level, title, desc in risks:
                clr  = "#f87171" if level == "HIGH" else "#fbbf24"
                bg   = "rgba(248,113,113,0.07)" if level == "HIGH" else "rgba(251,191,36,0.07)"
                icon = "🔴" if level == "HIGH" else "🟡"
                risk_html += (
                    f"<div style='background:{bg};border:1px solid {clr}44;"
                    f"border-left:3px solid {clr};border-radius:10px;"
                    f"padding:0.9rem 1.2rem;margin-bottom:0.7rem'>"
                    f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem'>"
                    f"<span>{icon}</span>"
                    f"<span style='font-family:Syne,sans-serif;font-size:0.85rem;font-weight:700;color:{clr}'>"
                    f"{level} RISK — {title}</span></div>"
                    f"<div style='font-size:0.83rem;color:var(--text);line-height:1.5'>{desc}</div>"
                    f"</div>"
                )
            st.markdown(risk_html, unsafe_allow_html=True)
            high_count = sum(1 for r in risks if r[0] == "HIGH")
            med_count  = sum(1 for r in risks if r[0] == "MEDIUM")
            st.markdown(
                f"<div style='font-size:0.8rem;color:var(--muted);margin-top:0.5rem;text-align:right'>"
                f"Found <span style='color:#f87171;font-weight:600'>{high_count} High</span> and "
                f"<span style='color:#fbbf24;font-weight:600'>{med_count} Medium</span> risk factors</div>",
                unsafe_allow_html=True
            )

    else:
        st.markdown("""
        <div style="background:rgba(56,189,248,0.05);border:1px dashed rgba(56,189,248,0.2);
                    border-radius:12px;padding:1.2rem 1.5rem;color:#64748b;font-size:0.9rem">
          ☝️ Enter a revenue target above to see the feasibility analysis
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="app-footer">Revenue Target Feasibility Analyzer · Built with Streamlit · Data processed locally</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 4 — USER ACTIVITY LOG
# ══════════════════════════════════════════════════════════
elif page == "👤  Activity Log":

    st.markdown("""
    <div class="animate-in" style="padding:1.5rem 0 0.5rem">
      <div class="hero-title">User Activity<br>Log</div>
      <p class="hero-sub">All users · Login history · Session tracking</p>
    </div>
    """, unsafe_allow_html=True)

    current_user = st.session_state.get("username", "")
    log          = load_activity()

    if not log:
        st.markdown(
            "<div style='background:rgba(56,189,248,0.05);border:1px dashed rgba(56,189,248,0.2);"
            "border-radius:12px;padding:1.2rem 1.5rem;color:#64748b;font-size:0.9rem'>"
            "No activity recorded yet. Login activity will appear here.</div>",
            unsafe_allow_html=True
        )
    else:
        # ── Overall Stats ─────────────────────────────
        total_users   = len(log)
        total_logins  = sum(v.get("login_count", 0) for v in log.values())
        most_active   = max(log, key=lambda u: log[u].get("login_count", 0))
        last_active   = max(log, key=lambda u: log[u].get("last_login", ""))

        section("📊", "Platform Overview")
        ov1, ov2, ov3, ov4 = st.columns(4)
        ov1.metric("Total Users",       str(total_users))
        ov2.metric("Total Logins",      str(total_logins))
        ov3.metric("Most Active User",  most_active)
        ov4.metric("Last Active User",  last_active)

        st.markdown("<div style='margin:1.5rem 0 0'></div>", unsafe_allow_html=True)

        # ── All Users Table ───────────────────────────
        section("👥", "All Users Activity")

        table_html = (
            "<div style='overflow-x:auto'>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.86rem'>"
            "<thead>"
            "<tr style='background:#1f2937'>"
            "<th style='padding:0.7rem 1rem;text-align:left;color:#38bdf8;font-family:Syne,sans-serif;"
            "font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1f2937'>#</th>"
            "<th style='padding:0.7rem 1rem;text-align:left;color:#38bdf8;font-family:Syne,sans-serif;"
            "font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1f2937'>Username</th>"
            "<th style='padding:0.7rem 1rem;text-align:center;color:#38bdf8;font-family:Syne,sans-serif;"
            "font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1f2937'>Total Logins</th>"
            "<th style='padding:0.7rem 1rem;text-align:left;color:#38bdf8;font-family:Syne,sans-serif;"
            "font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1f2937'>Last Login</th>"
            "<th style='padding:0.7rem 1rem;text-align:center;color:#38bdf8;font-family:Syne,sans-serif;"
            "font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1f2937'>Status</th>"
            "</tr>"
            "</thead><tbody>"
        )

        sorted_users = sorted(log.items(), key=lambda x: x[1].get("login_count", 0), reverse=True)
        for i, (uname, udata) in enumerate(sorted_users):
            is_current  = uname == current_user
            row_bg      = "#111827" if i % 2 == 0 else "#0d1117"
            badge_html  = (
                "<span style='background:rgba(52,211,153,0.12);border:1px solid rgba(52,211,153,0.3);"
                "border-radius:6px;padding:0.15rem 0.6rem;font-size:0.72rem;color:#34d399'>● Online</span>"
                if is_current else
                "<span style='background:rgba(100,116,139,0.12);border:1px solid rgba(100,116,139,0.2);"
                "border-radius:6px;padding:0.15rem 0.6rem;font-size:0.72rem;color:#64748b'>○ Offline</span>"
            )
            user_label = (
                f"<strong style='color:#38bdf8'>{uname}</strong>"
                f"<span style='margin-left:0.5rem;background:rgba(56,189,248,0.1);"
                f"border:1px solid rgba(56,189,248,0.25);border-radius:4px;"
                f"padding:0.1rem 0.4rem;font-size:0.7rem;color:#38bdf8'>You</span>"
                if is_current else
                f"<span style='color:#e2e8f0'>{uname}</span>"
            )
            logins     = udata.get("login_count", 0)
            last_login = udata.get("last_login", "Never")
            table_html += (
                f"<tr style='background:{row_bg};border-bottom:1px solid #1f2937'>"
                f"<td style='padding:0.75rem 1rem;color:#64748b'>{i+1}</td>"
                f"<td style='padding:0.75rem 1rem'>{user_label}</td>"
                f"<td style='padding:0.75rem 1rem;text-align:center;font-family:Syne,sans-serif;"
                f"font-weight:700;color:#38bdf8'>{logins}</td>"
                f"<td style='padding:0.75rem 1rem;color:#94a3b8'>{last_login}</td>"
                f"<td style='padding:0.75rem 1rem;text-align:center'>{badge_html}</td>"
                f"</tr>"
            )

        table_html += "</tbody></table></div>"
        st.markdown(table_html, unsafe_allow_html=True)

        # ── Current User Session History ──────────────
        st.markdown("<div style='margin:2rem 0 0'></div>", unsafe_allow_html=True)
        section("🕐", f"Your Recent Sessions — {current_user}")

        if current_user in log:
            sessions = log[current_user].get("sessions", [])[::-1]
            if sessions:
                session_html = ""
                for i, s in enumerate(sessions):
                    session_html += (
                        f"<div style='display:flex;align-items:center;gap:1rem;"
                        f"padding:0.7rem 1rem;background:var(--card);border:1px solid var(--border);"
                        f"border-radius:10px;margin-bottom:0.5rem'>"
                        f"<span style='font-family:Syne,sans-serif;font-size:0.75rem;font-weight:700;"
                        f"color:var(--muted);min-width:30px'>#{len(sessions)-i}</span>"
                        f"<span style='font-size:0.88rem;color:var(--text)'>{s}</span>"
                        f"{'<span style=\"background:rgba(52,211,153,0.12);border:1px solid rgba(52,211,153,0.3);border-radius:6px;padding:0.15rem 0.6rem;font-size:0.72rem;color:#34d399\">Current</span>' if i == 0 else ''}"
                        f"</div>"
                    )
                st.markdown(session_html, unsafe_allow_html=True)
            else:
                st.info("No session history yet.")
        else:
            st.info("No session history yet.")

    st.markdown('<div class="app-footer">Revenue Target Feasibility Analyzer · Built with Streamlit · Data processed locally</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════
elif page == "📊  Sales Analytics":

    st.markdown("""
    <div class="animate-in" style="padding:1.5rem 0 0.5rem">
      <div class="hero-title">Sales<br>Analytics</div>
      <p class="hero-sub">Trends · Categories · Products · Correlations</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Monthly Revenue",
        "🏷️  Category & Region",
        "📦  Top Products",
        "🔗  Correlations"
    ])

    # ── Tab 1: Monthly Revenue ────────────────
    with tab1:
        section("📈", "Monthly Revenue Trend")

        fig1, ax1 = plt.subplots(figsize=(13, 4))
        ax1.fill_between(monthly_revenue["Month_str"], monthly_revenue["Sales"],
                         alpha=0.08, color=ACCENT)
        ax1.plot(monthly_revenue["Month_str"], monthly_revenue["Sales"],
                 color=ACCENT, lw=2.2, marker='o', ms=4, zorder=3, label="Monthly Revenue")
        roll = monthly_revenue["Sales"].rolling(3, min_periods=1).mean()
        ax1.plot(monthly_revenue["Month_str"], roll,
                 color=ACCENT2, lw=1.8, ls='--', label="3-mo Rolling Avg", alpha=0.85)
        ax1.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8.5)
        plt.xticks(rotation=45, ha='right')
        style_fig(fig1, [ax1])
        st.pyplot(fig1, use_container_width=True)

        years = sorted(df["Year"].unique())
        if len(years) > 1:
            section("📅", "Year-over-Year Comparison")
            yoy = df.groupby(["Year", df["Order Date"].dt.month])["Sales"].sum().unstack(level=0)
            fig_y, ax_y = plt.subplots(figsize=(11, 3.5))
            palette = [ACCENT, ACCENT2, ACCENT3, WARN]
            for i, yr in enumerate(yoy.columns):
                ax_y.plot(yoy.index, yoy[yr], label=str(yr),
                          color=palette[i % len(palette)], lw=2.1,
                          marker='o', ms=3.5)
            ax_y.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8.5)
            ax_y.set_xticks(range(1,13))
            ax_y.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                                   'Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=8)
            style_fig(fig_y, [ax_y])
            st.pyplot(fig_y, use_container_width=True)

    # ── Tab 2: Category & Region ──────────────
    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            section("🏷️", "Sales by Category")
            cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=True)
            fig2, ax2 = plt.subplots(figsize=(6, 3.2))
            colors2 = [ACCENT, ACCENT2, ACCENT3][:len(cat_sales)]
            bars2 = ax2.barh(cat_sales.index, cat_sales.values,
                             color=colors2, height=0.45, zorder=3)
            ax2.bar_label(bars2, fmt='$%.0f', color=TEXT, fontsize=8, padding=5)
            ax2.set_xlabel("Sales ($)")
            style_fig(fig2, [ax2])
            ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v/1e3:.0f}k"))
            st.pyplot(fig2, use_container_width=True)

        with c2:
            section("🌍", "Profit by Region")
            reg_profit = df.groupby("Region")["Profit"].sum().sort_values(ascending=True)
            colors_r   = [ACCENT3 if v >= 0 else DANGER for v in reg_profit.values]
            fig3, ax3  = plt.subplots(figsize=(6, 3.2))
            bars3 = ax3.barh(reg_profit.index, reg_profit.values,
                             color=colors_r, height=0.45, zorder=3)
            ax3.bar_label(bars3, fmt='$%.0f', color=TEXT, fontsize=8, padding=5)
            ax3.set_xlabel("Profit ($)")
            style_fig(fig3, [ax3])
            ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v/1e3:.0f}k"))
            st.pyplot(fig3, use_container_width=True)

        if "Sub-Category" in df.columns:
            section("📂", "Sales & Profit by Sub-Category")
            sub = df.groupby("Sub-Category")[["Sales","Profit"]].sum().sort_values("Sales", ascending=False)
            fig_s, ax_s = plt.subplots(figsize=(13, 4))
            x = np.arange(len(sub)); w = 0.38
            ax_s.bar(x - w/2, sub["Sales"],  width=w, color=ACCENT,  label="Sales",  alpha=0.9, zorder=3)
            ax_s.bar(x + w/2, sub["Profit"], width=w, color=ACCENT2, label="Profit", alpha=0.9, zorder=3)
            ax_s.set_xticks(x)
            ax_s.set_xticklabels(sub.index, rotation=45, ha='right', fontsize=8)
            ax_s.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8.5)
            style_fig(fig_s, [ax_s])
            st.pyplot(fig_s, use_container_width=True)

    # ── Tab 3: Top Products ───────────────────
    with tab3:
        if "Product Name" in df.columns:
            col_n, _ = st.columns([1, 2])
            with col_n:
                n_top = st.slider("Top N Products", 5, 20, 10)
            section("🏆", f"Top {n_top} Products by Revenue")
            top_p = df.groupby("Product Name")["Sales"].sum().nlargest(n_top).sort_values()
            fig_p, ax_p = plt.subplots(figsize=(11, max(4, n_top * 0.42)))
            bar_colors = [plt.cm.cool(i / n_top) for i in range(n_top)]
            bars_p = ax_p.barh(top_p.index, top_p.values, color=bar_colors, height=0.6, zorder=3)
            ax_p.bar_label(bars_p, fmt='$%.0f', color=TEXT, fontsize=8, padding=5)
            ax_p.set_xlabel("Revenue ($)")
            style_fig(fig_p, [ax_p])
            ax_p.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v/1e3:.0f}k"))
            ax_p.tick_params(axis='y', labelsize=8)
            st.pyplot(fig_p, use_container_width=True)
        else:
            st.info("No 'Product Name' column found in the dataset.")

    # ── Tab 4: Correlations ───────────────────
    with tab4:
        section("🔗", "Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=np.number)
        fig4, ax4 = plt.subplots(figsize=(8, 5.5))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(
            numeric_cols.corr(), annot=True, fmt=".2f", cmap=cmap,
            ax=ax4, linewidths=0.6, linecolor=BORDER,
            annot_kws={"size": 9, "color": TEXT},
            cbar_kws={"shrink": 0.75}
        )
        ax4.set_facecolor(CARD)
        fig4.patch.set_facecolor("none")
        ax4.tick_params(colors=MUTED, labelsize=8.5)
        for spine in ax4.spines.values():
            spine.set_edgecolor(BORDER)
        cbar = ax4.collections[0].colorbar
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
        cbar.outline.set_edgecolor(BORDER)
        fig4.tight_layout(pad=1.5)
        st.pyplot(fig4, use_container_width=True)

    st.markdown('<div class="app-footer">Revenue Target Feasibility Analyzer · Built with Streamlit · Data processed locally</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — DOWNLOAD REPORT
# ══════════════════════════════════════════════════════════
elif page == "📥  Download Report":

    st.markdown("""
    <div class="animate-in" style="padding:1.5rem 0 0.5rem">
      <div class="hero-title">Download<br>Report</div>
      <p class="hero-sub">Export a clean PDF summary of your analysis</p>
    </div>
    """, unsafe_allow_html=True)

    section("📋", "Report Preview")

    # ── Compute all values needed ──────────────────
    total_sales   = df["Sales"].sum()
    total_profit  = df["Profit"].sum()
    total_orders  = df["Order ID"].nunique()
    profit_margin = (total_profit / total_sales * 100) if total_sales else 0
    avg_order_val = total_sales / total_orders if total_orders else 0
    date_min      = df["Order Date"].min().strftime("%b %Y")
    date_max      = df["Order Date"].max().strftime("%b %Y")

    # Monthly revenue & model
    mr = (df.groupby("Month")["Sales"].sum().reset_index().sort_values("Month"))
    mr["Time_Index"] = np.arange(len(mr))
    X_r = mr[["Time_Index"]]
    y_r = mr["Sales"]
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    mr_pipeline = Pipeline([
        ("poly",      PolynomialFeatures(degree=1, include_bias=False)),
        ("scaler",    StandardScaler()),
        ("regressor", LinearRegression())
    ])
    mr_pipeline.fit(X_r, y_r)
    mdl = mr_pipeline
    best_month  = mr.loc[mr["Sales"].idxmax(), "Month"]
    worst_month = mr.loc[mr["Sales"].idxmin(), "Month"]
    top_cat     = df.groupby("Category")["Sales"].sum().idxmax()
    top_region  = df.groupby("Region")["Sales"].sum().idxmax()
    top_cat_share = (df.groupby("Category")["Sales"].sum().max() / total_sales * 100)

    # Auto-pull values from Dashboard via session state
    rpt_forecast_months = st.session_state.get('forecast_period', 6)
    rpt_target          = st.session_state.get('target', 0.0)
    rpt_pred_tot        = st.session_state.get('predicted_total', float(
                            np.clip(mdl.predict(
                                np.arange(len(mr), len(mr) + rpt_forecast_months).reshape(-1,1)
                            ), 0, None).sum()))

    if rpt_target > 0:
        rpt_score    = (rpt_pred_tot / rpt_target) * 100
        rpt_gap      = rpt_pred_tot - rpt_target
        rpt_gap_pct  = (rpt_gap / rpt_target) * 100
        if rpt_score >= 100:
            rpt_verdict = "ACHIEVABLE"
            rpt_verdict_note = f"Forecast exceeds target by {abs(rpt_gap_pct):.1f}%"
        elif rpt_score >= 80:
            rpt_verdict = "CHALLENGING"
            rpt_verdict_note = f"You are {abs(rpt_gap_pct):.1f}% short - strong push needed"
        else:
            rpt_verdict = "UNREALISTIC"
            rpt_verdict_note = f"Forecast falls {abs(rpt_gap_pct):.1f}% below target"
    else:
        rpt_score        = None
        rpt_gap          = None
        rpt_gap_pct      = None
        rpt_verdict      = "NOT SET"
        rpt_verdict_note = "Go to Dashboard and enter a Revenue Target first"

    # Pre-compute all display vars — avoids inline conditionals breaking inside HTML f-strings
    target_display  = "${:,.0f}".format(rpt_target) if rpt_target > 0 else "Not set - go to Dashboard first"
    preview_target  = "${:,.0f}".format(rpt_target) if rpt_target > 0 else "Not Set"
    score_display   = "{:.1f}%".format(rpt_score) if rpt_score else "N/A"
    verdict_color   = '#34d399' if rpt_verdict == 'ACHIEVABLE' else ('#fbbf24' if rpt_verdict == 'CHALLENGING' else ('#f87171' if rpt_verdict == 'UNREALISTIC' else '#64748b'))
    verdict_icon    = 'OK' if rpt_verdict == 'ACHIEVABLE' else ('!!' if rpt_verdict == 'CHALLENGING' else ('X' if rpt_verdict == 'UNREALISTIC' else 'i'))
    verdict_emoji   = 'Achievable' if rpt_verdict == 'ACHIEVABLE' else ('Challenging' if rpt_verdict == 'CHALLENGING' else ('Unrealistic' if rpt_verdict == 'UNREALISTIC' else 'Not Set'))

    # Info banner
    st.markdown(f"""
    <div style="background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.2);
                border-radius:12px;padding:0.9rem 1.4rem;margin-bottom:1.5rem;
                display:flex;gap:2.5rem;flex-wrap:wrap;align-items:center">
      <span style="font-size:0.88rem;color:#94a3b8">
        📅 <strong style="color:#e2e8f0">Forecast Period:</strong> {rpt_forecast_months} months
      </span>
      <span style="font-size:0.88rem;color:#94a3b8">
        🎯 <strong style="color:#e2e8f0">Target:</strong> {target_display}
      </span>
      <span style="font-size:0.88rem;color:#94a3b8">
        🔮 <strong style="color:#e2e8f0">Predicted:</strong> ${rpt_pred_tot:,.0f}
      </span>
      <span style="font-size:0.75rem;color:#475569;margin-left:auto">
        ↑ Auto-pulled from Dashboard
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Preview card ───────────────────────────────
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:16px;padding:2rem;max-width:700px;margin:0 auto">

      <div style="text-align:center;margin-bottom:1.5rem">
        <div style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
                    background:linear-gradient(135deg,#38bdf8,#f472b6);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text">
          💹 Revenue Target Feasibility Analyzer
        </div>
        <div style="color:var(--muted);font-size:0.8rem;margin-top:0.3rem">
          Report generated on {datetime.now().strftime("%d %B %Y, %I:%M %p")}
        </div>
      </div>

      <div style="border-top:1px solid var(--border);padding-top:1.2rem;margin-bottom:1.2rem">
        <div style="font-family:Syne,sans-serif;font-size:0.85rem;font-weight:700;
                    color:var(--accent);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem">
          📊 Business KPIs
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem">
          <div style="background:#0d1117;border-radius:10px;padding:0.8rem">
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Total Sales</div>
            <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:var(--accent)">${total_sales/1e6:.2f}M</div>
          </div>
          <div style="background:#0d1117;border-radius:10px;padding:0.8rem">
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Total Profit</div>
            <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:var(--accent)">${total_profit/1e3:.0f}K</div>
          </div>
          <div style="background:#0d1117;border-radius:10px;padding:0.8rem">
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Profit Margin</div>
            <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:var(--accent)">{profit_margin:.1f}%</div>
          </div>
          <div style="background:#0d1117;border-radius:10px;padding:0.8rem">
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Total Orders</div>
            <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:var(--accent)">{total_orders:,}</div>
          </div>
          <div style="background:#0d1117;border-radius:10px;padding:0.8rem">
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Avg Order Value</div>
            <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:var(--accent)">${avg_order_val:,.0f}</div>
          </div>
          <div style="background:#0d1117;border-radius:10px;padding:0.8rem">
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Date Range</div>
            <div style="font-family:Syne,sans-serif;font-size:0.9rem;font-weight:700;color:var(--accent)">{date_min} → {date_max}</div>
          </div>
        </div>
      </div>

      <div style="border-top:1px solid var(--border);padding-top:1.2rem;margin-bottom:1.2rem">
        <div style="font-family:Syne,sans-serif;font-size:0.85rem;font-weight:700;
                    color:var(--accent);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem">
          🔎 Key Insights
        </div>
        <div style="display:flex;flex-direction:column;gap:0.5rem">
          <div style="background:#0d1117;border-radius:8px;padding:0.7rem 1rem;font-size:0.85rem;color:var(--text)">
            🏆 <strong>Top Category:</strong> {top_cat} contributes {top_cat_share:.1f}% of total revenue
          </div>
          <div style="background:#0d1117;border-radius:8px;padding:0.7rem 1rem;font-size:0.85rem;color:var(--text)">
            🌍 <strong>Strongest Region:</strong> {top_region} leads in revenue performance
          </div>
          <div style="background:#0d1117;border-radius:8px;padding:0.7rem 1rem;font-size:0.85rem;color:var(--text)">
            📈 <strong>Best Month:</strong> {best_month} | 📉 <strong>Worst Month:</strong> {worst_month}
          </div>
          <div style="background:#0d1117;border-radius:8px;padding:0.7rem 1rem;font-size:0.85rem;color:var(--text)">
            🤖 <strong>Model:</strong> Linear Regression | Trend slope: ${mdl.named_steps['regressor'].coef_[0]:,.0f}/month
          </div>
        </div>
      </div>

      <div style="border-top:1px solid var(--border);padding-top:1.2rem;margin-bottom:1.2rem">
        <div style="font-family:Syne,sans-serif;font-size:0.85rem;font-weight:700;
                    color:var(--accent);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem">
          🎯 Feasibility Analysis
        </div>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.8rem;margin-bottom:0.8rem">
          <div style="background:#0d1117;border-radius:10px;padding:0.8rem">
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Forecast Period</div>
            <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:var(--accent)">{rpt_forecast_months} months</div>
          </div>
          <div style="background:#0d1117;border-radius:10px;padding:0.8rem">
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Predicted Revenue</div>
            <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:var(--accent)">${rpt_pred_tot:,.0f}</div>
          </div>
          <div style="background:#0d1117;border-radius:10px;padding:0.8rem">
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Revenue Target</div>
            <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:var(--accent)">{preview_target}</div>
          </div>
          <div style="background:#0d1117;border-radius:10px;padding:0.8rem">
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Feasibility Score</div>
            <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:var(--accent)">{score_display}</div>
          </div>
        </div>
        <div style="background:#0d1117;border-radius:10px;padding:1rem;border-left:3px solid {verdict_color}">
          <div style="font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem">Verdict</div>
          <div style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;color:{verdict_color}">
            {verdict_emoji}
          </div>
          <div style="font-size:0.82rem;color:var(--muted);margin-top:0.25rem">{rpt_verdict_note}</div>
        </div>
      </div>

      <div style="border-top:1px solid var(--border);padding-top:1rem;text-align:center;
                  color:var(--muted);font-size:0.75rem">
        Built with Streamlit · Data processed locally · Private
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin:2rem 0 1rem'></div>", unsafe_allow_html=True)
    section("⬇️", "Generate & Download")

    # ── PDF Generation ─────────────────────────────
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Use DejaVu (built-in Unicode font in fpdf2)
        pdf.add_font("DejaVu", style="", fname="DejaVuSansCondensed.ttf", uni=True) if False else None

        def H(size, bold=False):
            pdf.set_font("Helvetica", "B" if bold else "", size)
        def safe(text):
            # Replace special chars not supported by Helvetica
            return (str(text)
                .replace("—", "-")
                .replace("–", "-")
                .replace("→", "->")
                .replace("←", "<-")
                .replace("°", " deg")
                .replace("’", "'")
                .replace("‘", "'")
                .replace("“", '"')
                .replace("”", '"'))

        # Header bar
        pdf.set_fill_color(6, 8, 16)
        pdf.rect(0, 0, 210, 42, 'F')
        H(20, bold=True)
        pdf.set_text_color(56, 189, 248)
        pdf.cell(0, 15, "", ln=True)
        pdf.cell(0, 10, "Revenue Target Feasibility Analyzer", ln=True, align="C")
        H(9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, f"Report generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}", ln=True, align="C")
        pdf.ln(8)

        # ── Section: Business KPIs ──
        pdf.set_fill_color(17, 24, 39)
        pdf.set_text_color(56, 189, 248)
        H(12, bold=True)
        pdf.cell(0, 8, "  BUSINESS KPIs", ln=True, fill=True)
        pdf.ln(3)

        kpis = [
            ("Total Sales",      f"${total_sales/1e6:.2f}M"),
            ("Total Profit",     f"${total_profit/1e3:.0f}K"),
            ("Total Orders",     f"{total_orders:,}"),
            ("Profit Margin",    f"{profit_margin:.1f}%"),
            ("Avg Order Value",  f"${avg_order_val:,.0f}"),
            ("Date Range",       f"{date_min} to {date_max}"),
        ]
        H(10)
        for i, (label, val) in enumerate(kpis):
            pdf.set_fill_color(13, 17, 23) if i % 2 == 0 else pdf.set_fill_color(17, 24, 39)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(95, 8, f"  {safe(label)}", fill=True)
            pdf.set_text_color(226, 232, 240)
            pdf.cell(95, 8, safe(val), ln=True, fill=True)
        pdf.ln(6)

        # ── Section: Key Insights ──
        pdf.set_fill_color(17, 24, 39)
        pdf.set_text_color(56, 189, 248)
        H(12, bold=True)
        pdf.cell(0, 8, "  KEY INSIGHTS", ln=True, fill=True)
        pdf.ln(3)

        health = "Healthy" if profit_margin > 15 else "Needs Improvement"
        insights = [
            f"Top Category: {top_cat} contributes {top_cat_share:.1f}% of total revenue",
            f"Strongest Region: {top_region} leads in revenue performance",
            f"Best Month: {best_month}  |  Worst Month: {worst_month}",
            f"Trend: ${mdl.named_steps['regressor'].coef_[0]:,.0f} increase per month (ML Pipeline)",
            f"Overall Profit Margin: {profit_margin:.1f}% - {health}",
        ]
        H(10)
        for ins in insights:
            pdf.set_text_color(52, 211, 153)
            pdf.cell(8, 7, ">>")
            pdf.set_text_color(226, 232, 240)
            pdf.cell(0, 7, safe(ins), ln=True)
        pdf.ln(6)

        # ── Section: Feasibility Analysis ──
        pdf.set_fill_color(17, 24, 39)
        pdf.set_text_color(56, 189, 248)
        H(12, bold=True)
        pdf.cell(0, 8, "  FEASIBILITY ANALYSIS", ln=True, fill=True)
        pdf.ln(3)

        feasibility_data = [
            ("Forecast Period",    f"{rpt_forecast_months} months"),
            ("Predicted Revenue",  f"${rpt_pred_tot:,.0f}"),
            ("Revenue Target",     f"${rpt_target:,.0f}" if rpt_target > 0 else "Not Set"),
            ("Feasibility Score",  f"{rpt_score:.1f}%" if rpt_score else "N/A"),
            ("Gap",                f"${rpt_gap:+,.0f}" if rpt_gap is not None else "N/A"),
        ]
        H(10)
        for i, (label, val) in enumerate(feasibility_data):
            pdf.set_fill_color(13, 17, 23) if i % 2 == 0 else pdf.set_fill_color(17, 24, 39)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(95, 8, f"  {safe(label)}", fill=True)
            pdf.set_text_color(226, 232, 240)
            pdf.cell(95, 8, safe(val), ln=True, fill=True)
        pdf.ln(3)

        # Verdict box
        if rpt_verdict == "ACHIEVABLE":
            pdf.set_fill_color(20, 50, 35)
            pdf.set_text_color(52, 211, 153)
        elif rpt_verdict == "CHALLENGING":
            pdf.set_fill_color(50, 40, 10)
            pdf.set_text_color(251, 191, 36)
        elif rpt_verdict == "UNREALISTIC":
            pdf.set_fill_color(50, 15, 15)
            pdf.set_text_color(248, 113, 113)
        else:
            pdf.set_fill_color(17, 24, 39)
            pdf.set_text_color(100, 116, 139)

        H(13, bold=True)
        verdict_icon = ">> " if rpt_verdict == "ACHIEVABLE" else ("!! " if rpt_verdict == "CHALLENGING" else ("XX " if rpt_verdict == "UNREALISTIC" else "-- "))
        pdf.cell(0, 10, f"  {verdict_icon}VERDICT: {rpt_verdict} - {safe(rpt_verdict_note)}", ln=True, fill=True)
        pdf.ln(6)

        # ── Section: Monthly Revenue Table ──
        pdf.set_fill_color(17, 24, 39)
        pdf.set_text_color(56, 189, 248)
        H(12, bold=True)
        pdf.cell(0, 8, "  MONTHLY REVENUE SUMMARY (Last 12 Months)", ln=True, fill=True)
        pdf.ln(3)

        last12 = mr.tail(12)
        H(9, bold=True)
        pdf.set_fill_color(31, 41, 55)
        pdf.set_text_color(56, 189, 248)
        pdf.cell(60, 7, "  Month", fill=True)
        pdf.cell(65, 7, "Revenue", fill=True)
        pdf.cell(65, 7, "vs Monthly Avg", fill=True, ln=True)

        avg_rev = mr["Sales"].mean()
        H(9)
        for i, (_, row) in enumerate(last12.iterrows()):
            diff = row["Sales"] - avg_rev
            diff_str = f"+${diff:,.0f}" if diff >= 0 else f"-${abs(diff):,.0f}"
            pdf.set_fill_color(13, 17, 23) if i % 2 == 0 else pdf.set_fill_color(17, 24, 39)
            pdf.set_text_color(226, 232, 240)
            pdf.cell(60, 7, f"  {safe(str(row['Month']))}", fill=True)
            pdf.cell(65, 7, f"${row['Sales']:,.0f}", fill=True)
            pdf.set_text_color(52, 211, 153) if diff >= 0 else pdf.set_text_color(248, 113, 113)
            pdf.cell(65, 7, diff_str, fill=True, ln=True)
        pdf.ln(8)

        # Footer
        pdf.set_text_color(100, 116, 139)
        H(8)
        pdf.cell(0, 6, "Revenue Target Feasibility Analyzer  |  Built with Streamlit  |  Data processed locally", align="C", ln=True)

        return bytes(pdf.output())

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating your report..."):
                try:
                    pdf_bytes = generate_pdf()
                    st.download_button(
                        label="⬇️ Download Report PDF",
                        data=pdf_bytes,
                        file_name=f"revenue_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Report ready! Click above to download.")
                except Exception as e:
                    st.error(f"Error generating PDF: {e}. Make sure fpdf2 is installed: pip install fpdf2")

    with col_info:
        st.markdown("""
        <div style="background:rgba(56,189,248,0.05);border:1px dashed rgba(56,189,248,0.2);
                    border-radius:12px;padding:1rem 1.2rem;font-size:0.85rem;color:#64748b;line-height:1.8">
          📄 <strong style="color:#e2e8f0">PDF includes:</strong><br>
          &nbsp;&nbsp;• Business KPIs summary<br>
          &nbsp;&nbsp;• Key insights from your data<br>
          &nbsp;&nbsp;• Feasibility score + verdict<br>
          &nbsp;&nbsp;• Monthly revenue table (last 12 months)<br>
          &nbsp;&nbsp;• vs Average performance per month
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="app-footer">Revenue Target Feasibility Analyzer · Built with Streamlit · Data processed locally</div>', unsafe_allow_html=True)
