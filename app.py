import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=" Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 50%, #130f40 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a4e 0%, #0f0c29 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #e2e2ff !important; }
section[data-testid="stSidebar"] .stSlider > label,
section[data-testid="stSidebar"] .stDateInput > label { color: #a78bfa !important; font-weight: 600 !important; }

/* Sidebar slider accent */
div[data-baseweb="slider"] [data-testid="stThumbValue"] { background: #7c3aed !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    box-shadow: 0 20px 60px rgba(79,70,229,0.4);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -40px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute; bottom: -80px; left: -30px;
    width: 280px; height: 280px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero h1 { color: #fff; font-size: 2.4rem; font-weight: 800; margin: 0 0 8px; }
.hero p  { color: rgba(255,255,255,0.82); font-size: 1.05rem; margin: 0; }

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
}

/* ── Metric tiles ── */
.metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
.metric-tile {
    flex: 1; min-width: 160px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}
.metric-tile:hover { transform: translateY(-3px); }
.metric-tile .accent { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.metric-tile .value  { font-size: 1.9rem; font-weight: 800; margin: 6px 0 4px; }
.metric-tile .label  { font-size: 0.82rem; color: rgba(255,255,255,0.55); }
.tile-purple .accent { color: #a78bfa; } .tile-purple .value { color: #ede9fe; }
.tile-blue   .accent { color: #60a5fa; } .tile-blue   .value { color: #dbeafe; }
.tile-green  .accent { color: #34d399; } .tile-green  .value { color: #d1fae5; }
.tile-orange .accent { color: #fb923c; } .tile-orange .value { color: #ffedd5; }

/* ── Section header ── */
.section-header {
    font-size: 1.1rem; font-weight: 700;
    color: #a78bfa; letter-spacing: 0.5px;
    margin-bottom: 16px; margin-top: 8px;
    display: flex; align-items: center; gap: 8px;
}

/* ── Insight banner ── */
.insight-high   { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.35);  border-radius: 12px; padding: 16px 20px; color: #fca5a5; }
.insight-mid    { background: rgba(251,191,36,0.12); border: 1px solid rgba(251,191,36,0.30); border-radius: 12px; padding: 16px 20px; color: #fde68a; }
.insight-low    { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.30); border-radius: 12px; padding: 16px 20px; color: #6ee7b7; }
.insight-banner span.icon { font-size: 1.5rem; margin-right: 10px; }
.insight-banner .msg { font-size: 0.95rem; font-weight: 500; }

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.4) !important;
    transition: all 0.2s !important;
    letter-spacing: 0.5px;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(124,58,237,0.55) !important;
}

/* Hide default streamlit header decoration but keep hamburger menu */
header { visibility: visible; background: transparent !important; border: none !important; padding: 0 !important; }
footer { visibility: hidden; }

/* Style header to show only hamburger menu */
header .stDecoration { display: none; }
header .stAppHeader { background: transparent !important; }

/* Make viewerBadge more subtle */
.viewerBadge_container { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('Energy_prediction_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('energy_dataset.csv')
    df['day'] = pd.to_datetime(df['day'])
    return df.sort_values(by='day')

model = load_model()
df    = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Energy Consumption Predictor")
    st.markdown("---")
    st.markdown("#### 🎛️ Input Parameters")

    input_date      = st.date_input("📅 Select Prediction Date", datetime.date.today())
    temperature     = st.slider("🌡️ Temperature (°C)", 0, 50, 25)
    humidity        = st.slider("💧 Humidity (%)", 0, 100, 60)
    windspeed       = st.slider("🌬️ Windspeed (km/h)", 0, 20, 5)
    appliance_count = st.slider("🔌 Appliance Count", 0, 50, 10)

    st.markdown("---")
    predict_btn = st.button("⚡ Predict Now")

    st.markdown("""
    <div style='margin-top:32px;padding:14px;background:rgba(255,255,255,0.05);border-radius:12px;border:1px solid rgba(255,255,255,0.08);'>
        <div style='color:#a78bfa;font-weight:700;margin-bottom:6px;font-size:0.82rem;letter-spacing:1px;text-transform:uppercase;'>About</div>
        <div style='color:rgba(255,255,255,0.6);font-size:0.80rem;line-height:1.6;'>
            ML-powered daily & weekly energy forecasting using weather, appliance usage, and historical lag features.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Derived features ──────────────────────────────────────────────────────────
input_date_dt  = pd.to_datetime(input_date)
day_of_week    = input_date_dt.weekday()
is_weekend     = 1 if day_of_week >= 5 else 0
day_names      = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

lag_1          = df['Energy_Consumption'].iloc[-1]
lag_2          = df['Energy_Consumption'].iloc[-2]
lag_7          = df['Energy_Consumption'].iloc[-7]
rolling_mean_3 = df['Energy_Consumption'].iloc[-3:].mean()
rolling_mean_7 = df['Energy_Consumption'].iloc[-7:].mean()

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>⚡ Energy Consumption Predictor</h1>
    <p>AI-powered household energy consumption forecasting · Daily & weekly insights</p>
</div>
""", unsafe_allow_html=True)

# ── Summary tiles (always visible) ───────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-tile tile-purple">
        <div class="accent">🌡 Temperature</div>
        <div class="value">{temperature}°</div>
        <div class="label">Celsius</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-tile tile-blue">
        <div class="accent">💧 Humidity</div>
        <div class="value">{humidity}%</div>
        <div class="label">Relative Humidity</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-tile tile-green">
        <div class="accent">🌬 Windspeed</div>
        <div class="value">{windspeed}</div>
        <div class="label">km/h</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-tile tile-orange">
        <div class="accent">🔌 Appliances</div>
        <div class="value">{appliance_count}</div>
        <div class="label">Active Devices</div>
    </div>""", unsafe_allow_html=True)

# ── Prediction logic ──────────────────────────────────────────────────────────
if predict_btn:
    input_features = [[
        temperature, humidity, windspeed, appliance_count,
        day_of_week, is_weekend, lag_1, lag_2, lag_7,
        rolling_mean_3, rolling_mean_7
    ]]
    prediction = model.predict(input_features)[0]

    # Weekly forecast
    future_preds   = []
    current_lag1   = lag_1
    current_lag2   = lag_2
    current_lag7   = lag_7

    for i in range(7):
        future_date = input_date_dt + datetime.timedelta(days=i)
        dow     = future_date.weekday()
        weekend = 1 if dow >= 5 else 0
        rm3     = (current_lag1 + current_lag2 + prediction) / 3
        rm7     = rolling_mean_7

        data    = [[temperature, humidity, windspeed, appliance_count,
                    dow, weekend, current_lag1, current_lag2, current_lag7, rm3, rm7]]
        pred    = model.predict(data)[0]
        future_preds.append(pred)

        current_lag2 = current_lag1
        current_lag1 = pred

    avg_energy   = np.mean(future_preds)
    date_labels  = [(input_date_dt + datetime.timedelta(days=i)).strftime("%a %d") for i in range(7)]

    # ── Result header ──
    st.markdown(f"""
    <div class="glass-card" style="text-align:center;border:1px solid rgba(124,58,237,0.40);background:rgba(124,58,237,0.12);">
        <div style="color:#c4b5fd;font-size:0.85rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🔋 Today's Prediction</div>
        <div style="font-size:3rem;font-weight:800;color:#ede9fe;">{prediction:.2f} <span style="font-size:1.4rem;font-weight:500;color:#a78bfa;">kWh</span></div>
        <div style="color:rgba(255,255,255,0.5);margin-top:6px;font-size:0.88rem;">{day_names[day_of_week]}, {input_date_dt.strftime('%d %B %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Weekly result tiles ──
    st.markdown('<div class="section-header">📈 7-Day Forecast</div>', unsafe_allow_html=True)
    cols = st.columns(7)
    palette = ["#818cf8","#a78bfa","#c084fc","#e879f9","#f472b6","#fb7185","#fb923c"]
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.10);
                        border-radius:12px;padding:14px 10px;text-align:center;">
                <div style="font-size:0.72rem;font-weight:700;color:{palette[i]};text-transform:uppercase;letter-spacing:0.5px;">{date_labels[i]}</div>
                <div style="font-size:1.25rem;font-weight:800;color:#f1f5f9;margin:6px 0;">{future_preds[i]:.1f}</div>
                <div style="font-size:0.70rem;color:rgba(255,255,255,0.45);">kWh</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart ──
    st.markdown('<div class="section-header">📊 Weekly Consumption Chart</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor('#10103a')
    ax.set_facecolor('#10103a')

    x = np.arange(7)
    gradient_colors = ["#818cf8","#a78bfa","#c084fc","#e879f9","#f472b6","#fb7185","#fb923c"]

    # Area shading
    ax.fill_between(x, future_preds, alpha=0.18, color='#a78bfa')

    # Line & dots
    ax.plot(x, future_preds, color='#a78bfa', linewidth=2.5, zorder=3)
    for xi, yi, c in zip(x, future_preds, gradient_colors):
        ax.scatter(xi, yi, color=c, s=90, zorder=4, edgecolors='#10103a', linewidths=1.5)

    # Labels on dots
    for xi, yi in zip(x, future_preds):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 12), ha='center', color='#e2e8f0', fontsize=8, fontweight='600')

    ax.set_xticks(x)
    ax.set_xticklabels(date_labels, color='#94a3b8', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y', colors='#94a3b8', labelsize=9)
    ax.spines[:].set_visible(False)
    ax.grid(axis='y', color=(1, 1, 1, 0.08), linestyle='--', linewidth=0.8)
    ax.set_ylabel("Energy (kWh)", color='#94a3b8', fontsize=9)

    st.pyplot(fig)
    plt.close(fig)

    # ── Insight banner ──
    st.markdown('<div class="section-header">💡 Smart Insight</div>', unsafe_allow_html=True)

    if avg_energy > 15:
        st.markdown(f"""
        <div class="insight-high insight-banner glass-card">
            <span class="icon">🔴</span>
            <span class="msg"><strong>High Usage Alert!</strong> Average weekly consumption is <strong>{avg_energy:.2f} kWh</strong>.
            Consider turning off unused appliances, using energy-efficient devices, and scheduling heavy loads during off-peak hours.</span>
        </div>""", unsafe_allow_html=True)
    elif avg_energy > 8:
        st.markdown(f"""
        <div class="insight-mid insight-banner glass-card">
            <span class="icon">🟡</span>
            <span class="msg"><strong>Moderate Consumption</strong> – Average: <strong>{avg_energy:.2f} kWh</strong>.
            Small adjustments in appliance usage or thermostat settings could bring noticeable savings.</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-low insight-banner glass-card">
            <span class="icon">🟢</span>
            <span class="msg"><strong>Excellent Efficiency!</strong> Average: <strong>{avg_energy:.2f} kWh</strong>.
            Your household is running at optimal energy levels. Keep it up!</span>
        </div>""", unsafe_allow_html=True)

    # ── Summary stats row ──
    st.markdown("<br>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="color:#a78bfa;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;">📊 Avg Daily</div>
            <div style="font-size:2rem;font-weight:800;color:#ede9fe;">{avg_energy:.2f} kWh</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="color:#60a5fa;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;">📈 Peak Day</div>
            <div style="font-size:2rem;font-weight:800;color:#dbeafe;">{max(future_preds):.2f} kWh</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="color:#34d399;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;">📉 Lowest Day</div>
            <div style="font-size:2rem;font-weight:800;color:#d1fae5;">{min(future_preds):.2f} kWh</div>
        </div>""", unsafe_allow_html=True)

else:
    # ── Placeholder when no prediction yet ──
    st.markdown("""
    <div class="glass-card" style="text-align:center;padding:60px 40px;">
        <div style="font-size:3.5rem;margin-bottom:16px;">⚡</div>
        <div style="color:#a78bfa;font-size:1.25rem;font-weight:700;margin-bottom:10px;">Ready to Forecast</div>
        <div style="color:rgba(255,255,255,0.45);font-size:0.92rem;max-width:420px;margin:0 auto;line-height:1.7;">
            Adjust the parameters in the sidebar and hit <strong style="color:#c4b5fd;">⚡ Predict Now</strong> 
            to get your personalized energy consumption prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)