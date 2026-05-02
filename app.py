"""
Livestock Health Monitor - Real-Time Dashboard
===============================================
Streamlit-based monitoring dashboard with live sensor simulation,
multi-animal tracking, alerts, and model explainability.

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import sys
import joblib
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Livestock Health Monitor",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.8; font-size: 0.95rem; }

    .status-healthy  { background: linear-gradient(135deg, #00b09b, #96c93d); color: white;
                        padding: 0.6rem 1.2rem; border-radius: 8px; font-weight: 600;
                        text-align: center; font-size: 1.1rem; }
    .status-sick      { background: linear-gradient(135deg, #e53935, #d32f2f); color: white;
                        padding: 0.6rem 1.2rem; border-radius: 8px; font-weight: 600;
                        text-align: center; font-size: 1.1rem; }
    .status-stressed  { background: linear-gradient(135deg, #ff8f00, #f9a825); color: white;
                        padding: 0.6rem 1.2rem; border-radius: 8px; font-weight: 600;
                        text-align: center; font-size: 1.1rem; }
    .status-inactive  { background: linear-gradient(135deg, #1565c0, #42a5f5); color: white;
                        padding: 0.6rem 1.2rem; border-radius: 8px; font-weight: 600;
                        text-align: center; font-size: 1.1rem; }

    .alert-box {
        background: linear-gradient(135deg, #ff1744, #d50000);
        color: white; padding: 1rem 1.5rem; border-radius: 10px;
        margin: 0.5rem 0; font-weight: 500;
        border-left: 5px solid #ffcdd2;
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        color: #0d47a1; padding: 0.8rem 1.2rem; border-radius: 8px;
        margin: 0.4rem 0; font-size: 0.9rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1rem;
        border: 1px solid #e9ecef; text-align: center;
    }
    .risk-low    { color: #2e7d32; font-weight: 700; }
    .risk-medium { color: #f57f17; font-weight: 700; }
    .risk-high   { color: #c62828; font-weight: 700; }

    div[data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 10px; padding: 0.8rem;
        border: 1px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: 0.5rem 1.5rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model + artifacts
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.joblib"))
    artifacts = joblib.load(os.path.join(MODELS_DIR, "preprocessing_artifacts.joblib"))
    meta = joblib.load(os.path.join(MODELS_DIR, "model_metadata.joblib"))
    model_size_kb = os.path.getsize(os.path.join(MODELS_DIR, "best_model.joblib")) / 1024
    return model, artifacts, meta, model_size_kb

model, artifacts, meta, model_size_kb = load_model()
scaler = artifacts["scaler"]
selector = artifacts["selector"]
all_features = artifacts["all_features"]
label_map = artifacts["label_map"]

# ---------------------------------------------------------------------------
# Cow profiles
# ---------------------------------------------------------------------------
COW_PROFILES = {
    "Bella (ID-1042)":   {"base_state": "healthy",  "emoji": "🐄", "age": "4 yrs", "breed": "Holstein"},
    "Daisy (ID-2087)":   {"base_state": "sick",     "emoji": "🐮", "age": "6 yrs", "breed": "Jersey"},
    "Luna (ID-3155)":    {"base_state": "stressed", "emoji": "🐄", "age": "3 yrs", "breed": "Angus"},
    "Rosie (ID-4201)":   {"base_state": "inactive", "emoji": "🐮", "age": "5 yrs", "breed": "Hereford"},
    "Clover (ID-5099)":  {"base_state": "healthy",  "emoji": "🐄", "age": "2 yrs", "breed": "Simmental"},
}

# ---------------------------------------------------------------------------
# Sensor simulator
# ---------------------------------------------------------------------------
def simulate_sensor_reading(base_state, t):
    """Generate realistic sensor data with temporal drift and noise."""
    phase = np.sin(t * 0.1)
    jitter = lambda s: np.random.normal(0, s)

    if base_state == "healthy":
        return {
            "acc_x": 0.18 + jitter(0.08) + 0.05 * phase,
            "acc_y": 0.12 + jitter(0.06),
            "acc_z": 9.81 + jitter(0.04),
            "temperature": 38.5 + jitter(0.2) + 0.1 * phase,
            "heart_rate": 65 + jitter(4) + 3 * phase,
            "respiration_rate": 24 + jitter(2),
            "rumination_rate": 55 + jitter(5),
            "hr_rolling_std": 6.5 + jitter(1),
            "hr_rolling_range": 18 + jitter(3),
            "activity_rolling_std": 0.25 + jitter(0.05),
            "activity_rolling_range": 0.9 + jitter(0.15),
            "movement_consistency": 40 + jitter(8),
        }
    elif base_state == "sick":
        fever = 40.5 + jitter(0.3) if np.random.random() > 0.3 else 37.0 + jitter(0.2)
        return {
            "acc_x": 0.03 + jitter(0.02),
            "acc_y": 0.02 + jitter(0.015),
            "acc_z": 9.80 + jitter(0.02),
            "temperature": fever,
            "heart_rate": 92 + jitter(10) + 5 * np.sin(t * 0.3),
            "respiration_rate": 45 + jitter(6),
            "rumination_rate": 12 + jitter(4),
            "hr_rolling_std": 18 + jitter(2),
            "hr_rolling_range": 74 + jitter(8),
            "activity_rolling_std": 0.28 + jitter(0.06),
            "activity_rolling_range": 1.2 + jitter(0.2),
            "movement_consistency": 200 + jitter(30),
        }
    elif base_state == "stressed":
        return {
            "acc_x": 0.40 + jitter(0.15) + 0.1 * np.sin(t * 0.5),
            "acc_y": 0.35 + jitter(0.12),
            "acc_z": 10.0 + jitter(0.1),
            "temperature": 39.5 + jitter(0.25),
            "heart_rate": 100 + jitter(6) + 4 * phase,
            "respiration_rate": 42 + jitter(4),
            "rumination_rate": 28 + jitter(6),
            "hr_rolling_std": 10 + jitter(1.2),
            "hr_rolling_range": 40 + jitter(5),
            "activity_rolling_std": 0.70 + jitter(0.08),
            "activity_rolling_range": 2.8 + jitter(0.3),
            "movement_consistency": 22 + jitter(5),
        }
    else:  # inactive
        return {
            "acc_x": 0.01 + jitter(0.005),
            "acc_y": 0.005 + jitter(0.003),
            "acc_z": 9.81 + jitter(0.005),
            "temperature": 37.9 + jitter(0.15),
            "heart_rate": 48 + jitter(2),
            "respiration_rate": 14 + jitter(1.5),
            "rumination_rate": 22 + jitter(3),
            "hr_rolling_std": 4.0 + jitter(0.5),
            "hr_rolling_range": 16 + jitter(2),
            "activity_rolling_std": 0.017 + jitter(0.003),
            "activity_rolling_range": 0.07 + jitter(0.01),
            "movement_consistency": 900 + jitter(50),
        }


def preprocess_and_predict(sensor_data):
    """Run the full inference pipeline."""
    ax, ay, az = sensor_data["acc_x"], sensor_data["acc_y"], sensor_data["acc_z"]
    hr = sensor_data["heart_rate"]
    temp = sensor_data["temperature"]
    resp = sensor_data["respiration_rate"]
    rum = sensor_data["rumination_rate"]

    sensor_data["acc_magnitude"] = np.sqrt(ax**2 + ay**2 + az**2)
    sensor_data["acc_variance"] = 0.01
    sensor_data["temp_deviation"] = abs(temp - 38.6)
    sensor_data["activity_index"] = abs(ax) + abs(ay) + abs(az - 9.81)
    sensor_data["resp_hr_ratio"] = resp / (hr + 0.001)
    sensor_data["acc_jerk"] = 0.0
    sensor_data["hr_temp_ratio"] = hr / temp
    sensor_data["activity_rumination"] = sensor_data["activity_index"] * rum
    sensor_data["resp_temp_interaction"] = resp * sensor_data["temp_deviation"]
    sensor_data["hr_instability"] = sensor_data["hr_rolling_std"] / (hr + 0.001)
    sensor_data["activity_consistency"] = sensor_data["activity_index"] * sensor_data["movement_consistency"]

    start = time.perf_counter()
    vec = np.array([[sensor_data[f] for f in all_features]])
    vec_sel = selector.transform(vec)
    vec_sc = scaler.transform(vec_sel)
    pred = model.predict(vec_sc)[0]
    proba = model.predict_proba(vec_sc)[0] if hasattr(model, "predict_proba") else None
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "state": int(pred),
        "label": label_map[pred],
        "confidence": float(max(proba)) if proba is not None else 0,
        "probabilities": {label_map[i]: float(p) for i, p in enumerate(proba)} if proba is not None else {},
        "latency_ms": latency_ms,
    }

# ---------------------------------------------------------------------------
# Risk & explanation helpers
# ---------------------------------------------------------------------------
def get_severity(label, confidence, sensor):
    if label in ("Healthy", "Inactive"):
        return "Low", "risk-low"
    temp_dev = abs(sensor["temperature"] - 38.6)
    hr_dev = abs(sensor["heart_rate"] - 65)
    score = temp_dev * 10 + hr_dev * 0.5
    if score > 30:
        return "High", "risk-high"
    elif score > 15:
        return "Medium", "risk-medium"
    return "Low", "risk-low"


def get_explanation(label, sensor):
    temp = sensor["temperature"]
    hr = sensor["heart_rate"]
    rum = sensor["rumination_rate"]
    resp = sensor["respiration_rate"]

    if label == "Healthy":
        return "All vitals within normal range. Regular activity and rumination observed."
    elif label == "Sick":
        reasons = []
        if temp > 39.5:
            reasons.append(f"High temperature ({temp:.1f} C) suggests fever")
        elif temp < 37.5:
            reasons.append(f"Low temperature ({temp:.1f} C) suggests hypothermia")
        if hr > 85:
            reasons.append(f"Elevated heart rate ({hr:.0f} BPM) indicates tachycardia")
        if rum < 20:
            reasons.append(f"Very low rumination ({rum:.0f}/min) indicates digestive distress")
        if resp > 40:
            reasons.append(f"Rapid breathing ({resp:.0f}/min) suggests respiratory distress")
        elif resp < 12:
            reasons.append(f"Depressed breathing ({resp:.0f}/min) suggests CNS depression")
        return " | ".join(reasons) if reasons else "Multiple abnormal vitals detected"
    elif label == "Stressed":
        reasons = []
        if hr > 90:
            reasons.append(f"Elevated HR ({hr:.0f} BPM) indicates acute stress response")
        if temp > 39.0:
            reasons.append(f"Mildly elevated temp ({temp:.1f} C) from cortisol response")
        if resp > 35:
            reasons.append(f"Panting ({resp:.0f} breaths/min) suggests heat stress")
        return " | ".join(reasons) if reasons else "Behavioral stress indicators detected"
    else:  # Inactive
        reasons = []
        if hr < 52:
            reasons.append(f"Low resting HR ({hr:.0f} BPM)")
        if rum < 25:
            reasons.append(f"Low rumination ({rum:.0f}/min) - likely sleeping")
        reasons.append("Minimal movement detected - normal resting behavior")
        return " | ".join(reasons)


def get_recommendation(label, severity):
    recs = {
        ("Sick", "High"):   "URGENT: Isolate animal immediately. Contact veterinarian. Check for infectious disease.",
        ("Sick", "Medium"): "Monitor closely for next 2 hours. Prepare veterinary consultation.",
        ("Sick", "Low"):    "Track vitals. Schedule routine veterinary check within 24 hours.",
        ("Stressed", "High"):   "Move animal to shaded area. Provide water. Reduce herd density.",
        ("Stressed", "Medium"): "Monitor environmental conditions. Ensure water access.",
        ("Stressed", "Low"):    "Minor stress detected. Continue normal monitoring.",
    }
    return recs.get((label, severity), "")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = {name: [] for name in COW_PROFILES}
if "tick" not in st.session_state:
    st.session_state.tick = 0
if "running" not in st.session_state:
    st.session_state.running = True

MAX_HISTORY = 20

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🐄 Livestock Health Monitoring System</h1>
    <p>Edge AI Real-Time Dashboard &nbsp;|&nbsp; On-Device Inference &nbsp;|&nbsp; Multi-Animal Tracking</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")
    refresh_rate = st.slider("Refresh Rate (sec)", 1, 5, 2)

    if st.button("Start / Resume" if not st.session_state.running else "Pause", use_container_width=True):
        st.session_state.running = not st.session_state.running

    if st.button("Reset History", use_container_width=True):
        st.session_state.history = {name: [] for name in COW_PROFILES}
        st.session_state.tick = 0

    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown(f"**Type:** {meta.get('name', 'Gradient Boosting')}")
    st.markdown(f"**Size:** {model_size_kb:.1f} KB")
    st.markdown(f"**Features:** {len(artifacts['selected_features'])}")
    st.markdown(f"**Classes:** 4")

    st.markdown("---")
    st.markdown("### Selected Features")
    for i, f in enumerate(artifacts["selected_features"], 1):
        st.markdown(f"`{i}. {f}`")

    st.markdown("---")
    st.markdown("### Legend")
    st.markdown("🟢 **Healthy** - Normal vitals")
    st.markdown("🔴 **Sick** - Abnormal vitals")
    st.markdown("🟡 **Stressed** - Elevated stress")
    st.markdown("🔵 **Inactive** - Resting/sleeping")

# ---------------------------------------------------------------------------
# Simulate one tick
# ---------------------------------------------------------------------------
t = st.session_state.tick
now = datetime.now()

all_results = {}
for cow_name, profile in COW_PROFILES.items():
    sensor = simulate_sensor_reading(profile["base_state"], t + hash(cow_name) % 100)
    result = preprocess_and_predict(sensor)
    sensor["timestamp"] = now
    result["sensor"] = sensor
    all_results[cow_name] = result

    hist = st.session_state.history[cow_name]
    hist.append({"timestamp": now, "sensor": sensor.copy(), "result": result.copy()})
    if len(hist) > MAX_HISTORY:
        st.session_state.history[cow_name] = hist[-MAX_HISTORY:]

# ---------------------------------------------------------------------------
# Fleet overview bar
# ---------------------------------------------------------------------------
status_counts = {"Healthy": 0, "Sick": 0, "Stressed": 0, "Inactive": 0}
for r in all_results.values():
    status_counts[r["label"]] += 1

cols_overview = st.columns(6)
cols_overview[0].metric("Total Animals", len(COW_PROFILES))
cols_overview[1].metric("🟢 Healthy", status_counts["Healthy"])
cols_overview[2].metric("🔴 Sick", status_counts["Sick"])
cols_overview[3].metric("🟡 Stressed", status_counts["Stressed"])
cols_overview[4].metric("🔵 Inactive", status_counts["Inactive"])

avg_latency = np.mean([r["latency_ms"] for r in all_results.values()])
cols_overview[5].metric("Avg Latency", f"{avg_latency:.2f} ms")

# ---------------------------------------------------------------------------
# Alerts banner
# ---------------------------------------------------------------------------
alerts = [(name, r) for name, r in all_results.items() if r["label"] in ("Sick", "Stressed")]
if alerts:
    for name, r in alerts:
        severity, _ = get_severity(r["label"], r["confidence"], r["sensor"])
        rec = get_recommendation(r["label"], severity)
        icon = "🚨" if r["label"] == "Sick" else "⚠️"
        st.markdown(
            f'<div class="alert-box">{icon} <b>ALERT - {name}</b>: '
            f'{r["label"]} detected (confidence {r["confidence"]*100:.0f}%, risk: {severity}) '
            f'&mdash; {rec}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Per-cow tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([f"{profile['emoji']} {name}" for name, profile in COW_PROFILES.items()])

for tab, (cow_name, profile) in zip(tabs, COW_PROFILES.items()):
    with tab:
        result = all_results[cow_name]
        sensor = result["sensor"]
        hist = st.session_state.history[cow_name]

        # Status row
        label = result["label"]
        css_class = f"status-{label.lower()}"
        severity, sev_class = get_severity(label, result["confidence"], sensor)

        col_status, col_info = st.columns([1, 2])
        with col_status:
            st.markdown(f'<div class="{css_class}">{label.upper()}</div>', unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {result['confidence']*100:.1f}%")
            st.markdown(f'**Risk Level:** <span class="{sev_class}">{severity}</span>', unsafe_allow_html=True)
            st.markdown(f"**Latency:** {result['latency_ms']:.2f} ms")
            st.markdown(f"**Breed:** {profile['breed']} &nbsp;|&nbsp; **Age:** {profile['age']}")

        with col_info:
            explanation = get_explanation(label, sensor)
            st.markdown(f'<div class="info-box"><b>Diagnosis:</b> {explanation}</div>', unsafe_allow_html=True)

            if label in ("Sick", "Stressed"):
                rec = get_recommendation(label, severity)
                if rec:
                    st.markdown(f'<div class="alert-box">💊 <b>Recommendation:</b> {rec}</div>', unsafe_allow_html=True)

        # Metrics row
        st.markdown("#### Live Sensor Readings")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Temperature", f"{sensor['temperature']:.1f} C",
                   delta=f"{sensor['temperature'] - 38.6:+.1f}" if abs(sensor['temperature'] - 38.6) > 0.5 else None)
        m2.metric("Heart Rate", f"{sensor['heart_rate']:.0f} BPM",
                   delta=f"{sensor['heart_rate'] - 65:+.0f}" if abs(sensor['heart_rate'] - 65) > 10 else None)
        m3.metric("Respiration", f"{sensor['respiration_rate']:.0f} /min")
        m4.metric("Rumination", f"{sensor['rumination_rate']:.0f} /min")
        m5.metric("Activity", f"{sensor['activity_rolling_std']:.3f}")

        # Charts
        if len(hist) >= 2:
            st.markdown("#### Vital Signs Trend (last 20 readings)")
            chart_col1, chart_col2 = st.columns(2)

            timestamps = [h["timestamp"] for h in hist]
            temps = [h["sensor"]["temperature"] for h in hist]
            hrs = [h["sensor"]["heart_rate"] for h in hist]
            resps = [h["sensor"]["respiration_rate"] for h in hist]
            rums = [h["sensor"]["rumination_rate"] for h in hist]

            with chart_col1:
                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(x=timestamps, y=temps, mode="lines+markers",
                    line=dict(color="#e53935", width=2), marker=dict(size=5), name="Temperature"))
                fig_temp.add_hline(y=38.6, line_dash="dash", line_color="green", annotation_text="Normal (38.6)")
                fig_temp.update_layout(title="Body Temperature (C)", height=250, margin=dict(l=40,r=20,t=40,b=30),
                    template="plotly_white", showlegend=False)
                st.plotly_chart(fig_temp, use_container_width=True)

            with chart_col2:
                fig_hr = go.Figure()
                fig_hr.add_trace(go.Scatter(x=timestamps, y=hrs, mode="lines+markers",
                    line=dict(color="#1565c0", width=2), marker=dict(size=5), name="Heart Rate"))
                fig_hr.add_hline(y=65, line_dash="dash", line_color="green", annotation_text="Normal (65)")
                fig_hr.update_layout(title="Heart Rate (BPM)", height=250, margin=dict(l=40,r=20,t=40,b=30),
                    template="plotly_white", showlegend=False)
                st.plotly_chart(fig_hr, use_container_width=True)

            chart_col3, chart_col4 = st.columns(2)
            with chart_col3:
                fig_resp = go.Figure()
                fig_resp.add_trace(go.Scatter(x=timestamps, y=resps, mode="lines+markers",
                    line=dict(color="#ff8f00", width=2), marker=dict(size=5), name="Respiration"))
                fig_resp.add_hline(y=24, line_dash="dash", line_color="green", annotation_text="Normal (24)")
                fig_resp.update_layout(title="Respiration Rate (/min)", height=250, margin=dict(l=40,r=20,t=40,b=30),
                    template="plotly_white", showlegend=False)
                st.plotly_chart(fig_resp, use_container_width=True)

            with chart_col4:
                fig_rum = go.Figure()
                fig_rum.add_trace(go.Scatter(x=timestamps, y=rums, mode="lines+markers",
                    line=dict(color="#2e7d32", width=2), marker=dict(size=5), name="Rumination"))
                fig_rum.add_hline(y=55, line_dash="dash", line_color="green", annotation_text="Normal (55)")
                fig_rum.update_layout(title="Rumination Rate (/min)", height=250, margin=dict(l=40,r=20,t=40,b=30),
                    template="plotly_white", showlegend=False)
                st.plotly_chart(fig_rum, use_container_width=True)

        # Probability distribution
        with st.expander("Model Confidence Distribution"):
            if result["probabilities"]:
                prob_df = pd.DataFrame([
                    {"Class": k, "Probability": v} for k, v in result["probabilities"].items()
                ])
                colors = {"Healthy": "#4caf50", "Sick": "#e53935", "Stressed": "#ff8f00", "Inactive": "#1565c0"}
                fig_prob = px.bar(prob_df, x="Class", y="Probability", color="Class",
                    color_discrete_map=colors, text_auto=".3f")
                fig_prob.update_layout(height=200, margin=dict(l=40,r=20,t=20,b=30),
                    showlegend=False, template="plotly_white", yaxis_range=[0,1])
                st.plotly_chart(fig_prob, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer performance info
# ---------------------------------------------------------------------------
st.markdown("---")
perf_cols = st.columns(4)
perf_cols[0].markdown(f"**Model:** {meta.get('name', 'Gradient Boosting')}")
perf_cols[1].markdown(f"**Model Size:** {model_size_kb:.1f} KB (limit: 50 MB)")
perf_cols[2].markdown(f"**Avg Latency:** {avg_latency:.2f} ms (limit: 100 ms)")
perf_cols[3].markdown(f"**Tick:** {st.session_state.tick} &nbsp;|&nbsp; **Animals:** {len(COW_PROFILES)}")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if st.session_state.running:
    st.session_state.tick += 1
    time.sleep(refresh_rate)
    st.rerun()
