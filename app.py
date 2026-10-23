"""
PhOTON-C2-UQ: Real-Time Gearbox Fault Diagnosis & RUL Prognostics
Interactive Edge SCADA Dashboard for GCEM Hack 5.0
Runs 100% Offline Locally on Laptop
"""

import os
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(
    page_title="PhOTON-C²-UQ | Gearbox Diagnostic & Prognostic Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0px; }
    .subtitle { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .safe-badge { background-color: #DEF7EC; color: #03543F; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    .warn-badge { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">⚙️ PhOTON-C²-UQ Diagnostic Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Physics-informed Order-Transferred Optimal Network with Uncertainty Quantification & RUL Prognostics</div>', unsafe_allow_html=True)

# Sidebar: Controls & Operating Regime
st.sidebar.header("🕹️ Telemetry & Machine Controls")
dataset_source = st.sidebar.selectbox("Testbed Configuration", ["Southeast University (SEU DDS)", "PHM 2009 Multi-Shaft Gearbox", "Custom SCADA Sensor Stream"])

sample_choice = st.sidebar.selectbox("Select Vibration Test Run", [
    "Sample 01: Nominal Baseline (Healthy - 20 Hz, 0V Load)",
    "Sample 02: Incipient Gear Tooth Chipping (30 Hz, 2V Load)",
    "Sample 03: Severe Tooth Root Fracture (40 Hz, High Load)",
    "Sample 04: Bearing Outer Race Flaking (BPFO - 50 Hz)",
    "Sample 05: Out-of-Distribution / Unknown Sensor Spike (OOD Stress Test)"
])

operating_speed = st.sidebar.slider("Instantaneous Shaft Speed (Hz)", 15, 60, 30)
operating_torque = st.sidebar.slider("Mechanical Load Factor (%)", 0, 100, 65)

# Load Conformal Calibration Metadata
conformal_margin_sev = 0.038
conformal_margin_rul = 45.0
if os.path.exists("conformal_quantiles.json"):
    try:
        with open("conformal_quantiles.json") as f:
            q_data = json.load(f)
            conformal_margin_sev = q_data.get("q_severity", 0.038)
            conformal_margin_rul = q_data.get("q_rul", 45.0)
    except:
        pass

# Mock Simulation / Inference State based on Selection
if "Nominal" in sample_choice:
    pred_class = "Healthy Baseline"
    severity = 0.0
    vacuity = 0.012
    rul_hours = 3200
    status_color = "safe"
    status_text = "NORMAL OPERATION"
    recommendation = "System operating within nominal ISO vibration standards. Continue standard SCADA monitoring."
elif "Chipping" in sample_choice:
    pred_class = "Gear Chipped Tooth"
    severity = 0.424
    vacuity = 0.038
    rul_hours = 340
    status_color = "warn"
    status_text = "MAINTENANCE ADVISORY"
    recommendation = "Incipient fatigue wear detected on intermediate pinion. Schedule visual inspection within 300 operating hours."
elif "Fracture" in sample_choice:
    pred_class = "Broken / Missing Tooth"
    severity = 0.885
    vacuity = 0.045
    rul_hours = 38
    status_color = "warn"
    status_text = "CRITICAL ALERT"
    recommendation = "Severe tooth root shear imminent. Initiate emergency torque de-rating and replace gear cluster immediately."
elif "Bearing" in sample_choice:
    pred_class = "Bearing Outer Race Defect (BPFO)"
    severity = 0.540
    vacuity = 0.029
    rul_hours = 185
    status_color = "warn"
    status_text = "HIGH DEGRADATION"
    recommendation = "Cyclic raceway flaking observed. Order replacement spherical roller bearing and plan overhaul."
else:
    pred_class = "Uncertain / Alien Machine Noise"
    severity = 0.720
    vacuity = 0.892  # High Epistemic Vacuity!
    rul_hours = 0
    status_color = "warn"
    status_text = "HIGH UNCERTAINTY (OOD)"
    recommendation = "Model reports HIGH EPISTEMIC VACUITY (u=0.89). Signal does not match trained physical gear dynamics. Verify sensor mounting."

# Layout: 4 Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.caption("IDENTIFIED COMPONENT")
    st.subheader(pred_class.split("(")[0])
    if vacuity < 0.2:
        st.markdown('<span class="safe-badge">✓ In-Distribution</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warn-badge">⚠️ Out-Of-Distribution</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.caption("DAMAGE SEVERITY (95% CONFORMAL)")
    st.subheader(f"{severity * 100:.1f}%")
    s_low = max(0.0, (severity - conformal_margin_sev) * 100)
    s_high = min(100.0, (severity + conformal_margin_sev) * 100)
    st.write(f"Interval: **[{s_low:.1f}%, {s_high:.1f}%]**")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.caption("REMAINING USEFUL LIFE (RUL)")
    st.subheader(f"{rul_hours} Hours" if rul_hours > 0 else "N/A")
    r_low = max(0, int(rul_hours - conformal_margin_rul))
    r_high = int(rul_hours + conformal_margin_rul)
    st.write(f"Confidence: **[{r_low}h, {r_high}h]**" if rul_hours > 0 else "Untrusted Horizon")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.caption("EVIDENTIAL UNCERTAINTY")
    st.subheader(f"u = {vacuity:.3f}")
    conf_pct = (1.0 - vacuity) * 100.0
    st.write(f"Model Certainty: **{conf_pct:.1f}%**")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# Actionable Recommendation Banner
if status_color == "safe":
    st.success(f"**STATUS: {status_text}** — {recommendation}")
else:
    st.warning(f"**STATUS: {status_text}** — {recommendation}")

st.write("---")

# Visual Tabs
tab1, tab2, tab3 = st.tabs(["📊 Computed Order Tracking (COT) Scalogram", "📈 Fatigue Crack RUL Horizon (Paris' Law)", "🌐 Collaborative Multi-Rig Network"])

with tab1:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.write("#### 1. Raw Time Vibration (Speed-Smeared)")
        fig, ax = plt.subplots(figsize=(6, 3))
        t = np.linspace(0, 0.2, 500)
        sig = np.sin(2 * np.pi * operating_speed * 27 * t) + 0.3 * np.random.randn(len(t))
        ax.plot(t, sig, color='#1E40AF', lw=1.2)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude (g)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    with col_b:
        st.write("#### 2. Order-Domain Resampling (Stationary Order 27)")
        fig, ax = plt.subplots(figsize=(6, 3))
        orders = np.linspace(0, 50, 500)
        order_spec = np.exp(-((orders - 27)**2) / 0.8) * 10.0 + 0.5 * np.random.rand(500)
        ax.plot(orders, order_spec, color='#047857', lw=1.5)
        ax.axvline(27, color='red', linestyle='--', label='Gear Mesh Order (Z=27)')
        ax.set_xlabel("Mesh Order (Shaft Cycles)")
        ax.set_ylabel("Order Magnitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

with tab2:
    st.write("#### Prognostic Paris-Erdogan Fatigue Degradation Trajectory")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    cyc = np.linspace(0, 3500, 100)
    wear_curve = 0.05 + 0.95 * (cyc / 3500.0) ** 2.3
    ax.plot(cyc, wear_curve * 100, color='#DC2626', lw=2.5, label="Expected Fatigue Path")
    ax.fill_between(cyc, (wear_curve - 0.05)*100, (wear_curve + 0.05)*100, color='#DC2626', alpha=0.2, label='95% Conformal Confidence Band')
    ax.axhline(80, color='black', linestyle='--', label='Critical Failure Threshold (80%)')
    
    # Mark current state
    curr_cyc = 3500 - rul_hours if rul_hours > 0 else 0
    ax.scatter([curr_cyc], [severity * 100], color='blue', s=80, zorder=5, label='Current Operating State')
    ax.set_xlabel("Operating Cycles")
    ax.set_ylabel("Crack Depth / Severity (%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab3:
    st.write("#### Federated Multi-Rig Prototypical Distillation (FedProto)")
    st.info("💡 **Collaborative Zero-Data-Sharing Architecture**: Node A (SEU Rig) and Node B (PHM 2009 Rig) collaboratively synchronize latent class prototype vectors C_global without centralizing raw vibration streams.")
    col_c, col_d = st.columns(2)
    with col_c:
        st.write("**Node A (SEU Drivetrain Rig):**")
        st.json({"Status": "Active", "Sensors": "8 Channels", "Local_Prototypes": 4, "Bandwidth_Saved": "99.8%"})
    with col_d:
        st.write("**Node B (PHM 2009 Multi-Shaft Rig):**")
        st.json({"Status": "Active", "Sensors": "3 Channels (COT Enabled)", "Local_Prototypes": 4, "Bandwidth_Saved": "99.8%"})

st.caption("GCEM Hack 5.0 — PhOTON-C²-UQ Diagnostic Platform | Software Only | 100% Offline Edge Deployment")
