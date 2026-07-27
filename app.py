import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wattline | Energy Load Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────
# THEME — Cyber-Professional Neon Dashboard
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700;800;900&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-base: #05050A;
    --panel-bg: rgba(13, 16, 27, 0.65);
    --panel-border: rgba(0, 240, 255, 0.15);
    --panel-hover: rgba(0, 240, 255, 0.4);
    --text-main: #FFFFFF;
    --text-muted: #8B9BB4;
    --neon-cyan: #00F0FF;
    --neon-purple: #9D00FF;
    --danger: #FF0055;
    --glass-blur: blur(12px);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-base);
}

.stApp {
    background: radial-gradient(circle at top right, rgba(157, 0, 255, 0.05), transparent 40%),
                radial-gradient(circle at bottom left, rgba(0, 240, 255, 0.05), transparent 40%);
    background-color: var(--bg-base);
    color: var(--text-main);
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1140px;
}

/* ---- Animations ---- */
@keyframes floatUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes neonPulse {
    0% { text-shadow: 0 0 10px rgba(0, 240, 255, 0.2); }
    50% { text-shadow: 0 0 20px rgba(0, 240, 255, 0.6), 0 0 30px rgba(0, 240, 255, 0.4); }
    100% { text-shadow: 0 0 10px rgba(0, 240, 255, 0.2); }
}

@keyframes borderGlow {
    0% { box-shadow: 0 0 5px rgba(0, 240, 255, 0.1); }
    50% { box-shadow: 0 0 15px rgba(0, 240, 255, 0.3), inset 0 0 10px rgba(0, 240, 255, 0.1); }
    100% { box-shadow: 0 0 5px rgba(0, 240, 255, 0.1); }
}

/* ---- Hero ---- */
.wl-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 35px 40px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(13, 16, 27, 0.8) 0%, rgba(5, 5, 10, 0.9) 100%);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--panel-border);
    margin-bottom: 30px;
    animation: floatUp 0.8s ease-out forwards;
    position: relative;
    overflow: hidden;
}

.wl-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 4px; height: 100%;
    background: linear-gradient(to bottom, var(--neon-cyan), var(--neon-purple));
    box-shadow: 0 0 15px var(--neon-cyan);
}

.wl-hero-title {
    font-family: 'Montserrat', sans-serif;
    font-weight: 900;
    font-size: 2.4rem;
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.1;
    text-transform: uppercase;
    background: linear-gradient(90deg, #FFFFFF 0%, var(--text-muted) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.wl-hero-title span {
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: neonPulse 3s infinite alternate;
}

.wl-hero-sub {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-top: 10px;
    max-width: 550px;
    line-height: 1.6;
}

.wl-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--neon-cyan);
    background: rgba(0, 240, 255, 0.05);
    border: 1px solid rgba(0, 240, 255, 0.3);
    padding: 10px 20px;
    border-radius: 8px;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
    white-space: nowrap;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ---- Panels ---- */
.wl-panel {
    background: var(--panel-bg);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 28px;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    animation: floatUp 0.8s ease-out 0.2s forwards;
    opacity: 0;
}

.wl-panel:hover {
    border-color: var(--panel-hover);
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 240, 255, 0.1);
}

.wl-panel-title {
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--text-main);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: linear-gradient(90deg, var(--neon-cyan), #FFFFFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.wl-panel-sub {
    color: var(--text-muted);
    font-size: 0.88rem;
    margin-bottom: 24px;
    line-height: 1.5;
}

/* ---- Digital readout ---- */
.wl-meter {
    background: rgba(5, 5, 10, 0.8);
    border: 1px solid rgba(0, 240, 255, 0.2);
    border-radius: 10px;
    padding: 28px;
    position: relative;
    box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.8);
    animation: borderGlow 4s infinite alternate;
}

.wl-meter-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 12px;
}

.wl-meter-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 3.2rem;
    color: var(--neon-cyan);
    line-height: 1;
    text-shadow: 0 0 20px rgba(0, 240, 255, 0.4);
}

.wl-meter-unit {
    font-size: 1.2rem;
    color: var(--text-muted);
    margin-left: 10px;
    font-weight: 400;
    text-shadow: none;
}

/* ---- Chips ---- */
.wl-chip-row {
    display: flex;
    gap: 15px;
    margin-top: 25px;
    flex-wrap: wrap;
}

.wl-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    border-radius: 8px;
    padding: 12px 18px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.03);
    color: var(--text-muted);
    flex: 1;
    min-width: 170px;
    display: flex;
    justify-content: space-between;
    transition: all 0.3s ease;
}

.wl-chip:hover {
    background: rgba(157, 0, 255, 0.1);
    border-color: rgba(157, 0, 255, 0.4);
    color: var(--text-main);
}

.wl-chip b {
    color: var(--neon-purple);
    text-shadow: 0 0 10px rgba(157, 0, 255, 0.5);
}

/* ---- Buttons ---- */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #0057FF 0%, var(--neon-cyan) 100%) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 800 !important;
    padding: 0.7rem 1.8rem !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.95rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    width: 100%;
    position: relative;
    overflow: hidden;
    z-index: 1;
    transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    box-shadow: 0 4px 15px rgba(0, 240, 255, 0.2) !important;
}

.stButton > button::before,
.stFormSubmitButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    transition: all 0.5s ease;
    z-index: -1;
}

.stButton > button:hover::before,
.stFormSubmitButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 25px rgba(0, 240, 255, 0.5), 0 0 15px var(--neon-cyan) !important;
    color: #FFFFFF !important;
}

/* ---- Inputs ---- */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="number-input"] > div {
    background: rgba(0, 0, 0, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: var(--text-main) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="number-input"] > div:focus-within {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.2) !important;
}

label {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 15px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 30px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Montserrat', sans-serif;
    color: var(--text-muted);
    font-weight: 700;
    font-size: 0.95rem;
    padding: 12px 20px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px 8px 0 0;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    color: var(--neon-cyan) !important;
    background: rgba(0, 240, 255, 0.05) !important;
    border-bottom-color: var(--neon-cyan) !important;
    box-shadow: inset 0 -3px 10px rgba(0, 240, 255, 0.1);
}

.stTabs [aria-selected="true"] [data-testid="stMarkdownContainer"] {
    color: var(--neon-cyan) !important;
    text-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
}

/* ---- Metrics ---- */
div[data-testid="stMetric"] {
    background: rgba(13, 16, 27, 0.65);
    backdrop-filter: var(--glass-blur);
    border: 1px solid rgba(157, 0, 255, 0.2);
    border-radius: 12px;
    padding: 24px;
    transition: all 0.3s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: rgba(157, 0, 255, 0.6);
    box-shadow: 0 5px 20px rgba(157, 0, 255, 0.15);
    transform: translateY(-2px);
}

div[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-family: 'Montserrat', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stMetricValue"] {
    color: var(--text-main) !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem !important;
    font-weight: 700;
    background: linear-gradient(90deg, #FFFFFF, var(--neon-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ---- Footer ---- */
.wl-foot {
    color: var(--text-muted);
    font-size: 0.75rem;
    text-align: center;
    margin-top: 60px;
    margin-bottom: 24px;
    letter-spacing: 0.15em;
    font-family: 'JetBrains Mono', monospace;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    padding-top: 24px;
    text-transform: uppercase;
}

.wl-foot span {
    color: var(--neon-cyan);
    font-weight: 700;
}

/* ---- Empty state ---- */
.wl-empty {
    height: 100%;
    min-height: 340px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 40px 24px;
    border: 1px dashed rgba(0, 240, 255, 0.3);
    background: rgba(0, 240, 255, 0.02);
}

.wl-empty-text {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.05rem;
    color: var(--text-muted);
    line-height: 1.8;
}

.wl-empty-text b {
    color: var(--neon-cyan);
    font-weight: 700;
    text-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# DATA / MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    "Square Footage", "Number of Occupants", "Appliances Used",
    "Average Temperature", "Building Type_Industrial",
    "Building Type_Residential", "Day of Week_Weekend",
]
TARGET_COL = "Energy Consumption"

@st.cache_resource(show_spinner=False)
def load_builtin_artifacts():
    model = joblib.load("linear_regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data(show_spinner=False)
def load_frames():
    train_df = pd.read_csv("train_energy_data.csv")
    test_df = pd.read_csv("test_energy_data.csv")
    train_enc = pd.get_dummies(train_df, drop_first=True, dtype=int)
    test_enc = pd.get_dummies(test_df, drop_first=True, dtype=int)
    train_enc, test_enc = train_enc.align(test_enc, join="inner", axis=1)
    return train_df, test_df, train_enc, test_enc

class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.05, iterations=1500):
        self.lr = learning_rate
        self.iterations = iterations
        self.theta = None
        self.cost_history = []

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        for _ in range(self.iterations):
            predictions = X.dot(self.theta)
            errors = predictions - y
            gradients = (1 / m) * X.T.dot(errors)
            self.theta -= self.lr * gradients
            cost = (1 / (2 * m)) * np.sum(errors ** 2)
            self.cost_history.append(cost)

    def predict(self, X):
        return X.dot(self.theta)

@st.cache_resource(show_spinner=False)
def train_scratch_model(_scaler):
    _, _, train_enc, test_enc = load_frames()
    X_train = train_enc[FEATURE_ORDER].values
    y_train = train_enc[TARGET_COL].values
    X_test = test_enc[FEATURE_ORDER].values
    y_test = test_enc[TARGET_COL].values
    X_train_scaled = _scaler.transform(X_train)
    X_test_scaled = _scaler.transform(X_test)
    X_train_b = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
    X_test_b = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]
    model = LinearRegressionFromScratch(learning_rate=0.05, iterations=1500)
    model.fit(X_train_b, y_train)
    return model, X_test_b, y_test

def evaluate(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

# ──────────────────────────────────────────────────────────────────────────
# BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────────
missing = [f for f in [
    "linear_regression_model.pkl", "scaler.pkl",
    "train_energy_data.csv", "test_energy_data.csv",
] if not os.path.exists(f)]

if missing:
    st.markdown(f"""
    <div class="wl-panel" style="border-left: 4px solid var(--danger);">
        <div class="wl-panel-title" style="color:var(--danger);">System Alert: Missing Artifacts</div>
        <div class="wl-panel-sub">The application requires the following files in the working directory to initialize:</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:var(--text-main); line-height:1.8; background:rgba(0,0,0,0.5); padding:16px; border-radius:6px; border:1px solid rgba(255,0,85,0.3);">
            {"<br>".join(missing)}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

sk_model, scaler = load_builtin_artifacts()
train_df, test_df, train_enc, test_enc = load_frames()
scratch_model, X_test_scratch, y_test = train_scratch_model(scaler)
sk_pred_test = sk_model.predict(scaler.transform(test_enc[FEATURE_ORDER].values))
scratch_pred_test = scratch_model.predict(X_test_scratch)
sk_metrics = evaluate(y_test, sk_pred_test)
scratch_metrics = evaluate(y_test, scratch_pred_test)
TRAIN_MIN = train_df[TARGET_COL].min()
TRAIN_MAX = train_df[TARGET_COL].max()

# ──────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="wl-hero">
    <div>
        <div class="wl-hero-title">WATT<span>LINE</span></div>
        <div class="wl-hero-sub">Predict building energy loads utilizing parallel comparative linear models based on environmental and structural specifications.</div>
    </div>
    <div class="wl-badge">SYS_ACCURACY // {sk_metrics['R2']:.3f} R²</div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_lab = st.tabs(["PREDICTION ENGINE", "MODEL ANALYTICS"])

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICT
# ──────────────────────────────────────────────────────────────────────────
with tab_predict:
    col_form, col_out = st.columns([1, 1.2], gap="large")

    with col_form:
        st.markdown('<div class="wl-panel">', unsafe_allow_html=True)
        st.markdown('<div class="wl-panel-title">Facility Specifications</div>', unsafe_allow_html=True)
        st.markdown('<div class="wl-panel-sub">Input parameters mirror the training preprocessing pipeline architecture.</div>', unsafe_allow_html=True)

        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            with c1:
                sq_footage = st.number_input("Square Footage", min_value=100, value=1000)
                occupants = st.number_input("Occupancy Count", min_value=0, value=4)
                temp = st.number_input("Avg Temperature (°C)", value=25.0)
            with c2:
                appliances = st.number_input("Active Appliances", min_value=0, value=5)
                building_type = st.selectbox("Facility Classification", ["Residential", "Commercial", "Industrial"])
                day_of_week = st.selectbox("Operational Period", ["Weekday", "Weekend"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("INITIALIZE PREDICTION")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_out:
        if submitted:
            b_industrial = 1 if building_type == "Industrial" else 0
            b_residential = 1 if building_type == "Residential" else 0
            d_weekend = 1 if day_of_week == "Weekend" else 0

            features = np.array([[
                sq_footage, occupants, appliances, temp,
                b_industrial, b_residential, d_weekend,
            ]])
            features_scaled = scaler.transform(features)
            sk_prediction = sk_model.predict(features_scaled)[0]
            features_b = np.c_[np.ones(1), features_scaled]
            scratch_prediction = scratch_model.predict(features_b)[0]
            delta = sk_prediction - scratch_prediction

            st.markdown(f"""
            <div class="wl-panel" style="padding:0; overflow:hidden; border:none; background:transparent; animation-delay:0s;">
                <div class="wl-meter">
                    <div class="wl-meter-label">Projected Load Requirement (Scikit-Learn)</div>
                    <div class="wl-meter-value">{sk_prediction:,.0f}<span class="wl-meter-unit">kWh</span></div>
                    <div class="wl-chip-row">
                        <div class="wl-chip"><span>Algorithmic Base</span> <b>{scratch_prediction:,.0f} kWh</b></div>
                        <div class="wl-chip"><span>Variance</span> <b>{delta:+.1f} kWh</b></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+delta",
                value=sk_prediction,
                delta={
                    "reference": scratch_prediction,
                    "increasing": {"color": "#00F0FF"},
                    "decreasing": {"color": "#00F0FF"},
                    "font": {"size": 15}
                },
                gauge={
                    "axis": {
                        "range": [0, TRAIN_MAX * 1.1],
                        "tickcolor": "#8B9BB4",
                        "tickfont": {"color": "#8B9BB4", "size": 12}
                    },
                    "bar": {"color": "#9D00FF", "thickness": 0.7},
                    "bgcolor": "#05050A",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, TRAIN_MIN + (TRAIN_MAX - TRAIN_MIN) * 0.33], "color": "rgba(0, 240, 255, 0.08)"},
                        {"range": [TRAIN_MIN + (TRAIN_MAX - TRAIN_MIN) * 0.33, TRAIN_MIN + (TRAIN_MAX - TRAIN_MIN) * 0.66], "color": "rgba(157, 0, 255, 0.08)"},
                        {"range": [TRAIN_MIN + (TRAIN_MAX - TRAIN_MIN) * 0.66, TRAIN_MAX * 1.1], "color": "rgba(255, 0, 85, 0.08)"},
                    ],
                },
            ))
            fig.update_layout(
                height=280,
                margin=dict(l=40, r=40, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#FFFFFF", "family": "JetBrains Mono"},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        else:
            st.markdown("""
            <div class="wl-panel wl-empty" style="animation-delay: 0.1s;">
                <div class="wl-empty-text">
                    Awaiting facility parameters.<br>
                    Configure specifications and select<br>
                    <b>INITIALIZE PREDICTION</b> to execute load calculation.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL LAB
# ──────────────────────────────────────────────────────────────────────────
with tab_lab:
    st.markdown('<div class="wl-panel-title" style="font-size:1.2rem; margin-bottom:4px;">Comparative Architecture Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="wl-panel-sub" style="margin-bottom:24px;">Evaluation of standard implementation versus foundational gradient descent algorithm.</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("MSE Variance (Scikit)", f"{sk_metrics['MSE']:,.1f}", f"{sk_metrics['MSE']-scratch_metrics['MSE']:,.1f} vs base")
    m2.metric("MAE Variance (Scikit)", f"{sk_metrics['MAE']:,.1f}", f"{sk_metrics['MAE']-scratch_metrics['MAE']:,.1f} vs base")
    m3.metric("R² Efficiency (Scikit)", f"{sk_metrics['R2']:.4f}", f"{sk_metrics['R2']-scratch_metrics['R2']:+.4f} vs base")

    st.write("")

    col_bar, col_cost = st.columns(2, gap="large")

    with col_bar:
        st.markdown('<div class="wl-panel" style="animation-delay: 0.3s;">', unsafe_allow_html=True)
        st.markdown('<div class="wl-panel-title">Performance Metrics Validation</div>', unsafe_allow_html=True)

        fig_bar = go.Figure()
        metrics_names = ["MSE", "MAE", "R2"]
        fig_bar.add_trace(go.Bar(
            name="Scikit-Learn Implementation",
            x=metrics_names,
            y=[sk_metrics[m] for m in metrics_names],
            marker_color="#00F0FF",
            marker_line_width=0
        ))
        fig_bar.add_trace(go.Bar(
            name="Foundational Algorithm",
            x=metrics_names,
            y=[scratch_metrics[m] for m in metrics_names],
            marker_color="#9D00FF",
            marker_line_width=0
        ))
        fig_bar.update_layout(
            barmode="group",
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#FFFFFF", "size": 12},
            legend=dict(orientation="h", y=1.15, x=0, font=dict(size=12)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", tickfont=dict(size=11)),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", tickfont=dict(size=12)),
            bargap=0.25,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cost:
        st.markdown('<div class="wl-panel" style="animation-delay: 0.4s;">', unsafe_allow_html=True)
        st.markdown('<div class="wl-panel-title">Gradient Descent Convergence</div>', unsafe_allow_html=True)

        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            y=scratch_model.cost_history,
            mode="lines",
            line=dict(color="#00F0FF", width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 240, 255, 0.15)",
        ))
        fig_cost.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#FFFFFF", "size": 12},
            yaxis=dict(title="System Cost (MSE/2)", gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", title_font=dict(size=12)),
            xaxis=dict(title="Iteration Cycle", gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", title_font=dict(size=12)),
        )
        st.plotly_chart(fig_cost, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("SYSTEM PREPROCESSING PIPELINE"):
        st.markdown("""
- **Data Encoding** — `Facility Classification` and `Operational Period` variables are mapped via one-hot encoding (`drop_first=True`). Matrix alignment guarantees structural parity between datasets.
- **Data Normalization** — Features undergo standardization via `StandardScaler`, anchored exclusively to training subset parameters to prevent data leakage.
- **Vector Structure** — Square Footage · Occupancy Count · Active Appliances · Avg Temperature · Classification_Industrial · Classification_Residential · Operational_Weekend.
- **State Persistence** — Primary models and transformation matrices are serialized utilizing `joblib` for immediate deployment availability.
        """)

st.markdown("""
<div class="wl-foot">WATTLINE SYSTEM ARCHITECTURE // <span>SCIKIT-LEARN</span> & <span>BASE ALGORITHMIC MODELS</span></div>
""", unsafe_allow_html=True)