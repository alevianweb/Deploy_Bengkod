import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="◈",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 3rem; max-width: 1200px; }
.stApp { background: #0d0d0d; color: #e0e0e0; }

.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 0 1.5rem 0; border-bottom: 1px solid #1a1a1a; margin-bottom: 2rem;
}
.topbar-logo { font-size: 1rem; font-weight: 600; color: #fff; letter-spacing: -0.02em; }
.topbar-sub  { font-size: 0.72rem; color: #444; margin-top: 0.1rem; }

.stTabs [data-baseweb="tab-list"] { background: transparent !important; gap: 0 !important; border-bottom: 1px solid #1a1a1a !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #555 !important; font-size: 0.8rem !important; font-weight: 400 !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; padding: 0.65rem 1.4rem !important; border-radius: 0 !important; border: none !important; }
.stTabs [aria-selected="true"] { color: #fff !important; border-bottom: 1px solid #fff !important; font-weight: 500 !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

h1 { font-size: 1.5rem !important; font-weight: 600 !important; color: #fff !important; letter-spacing: -0.02em !important; margin-bottom: 0.3rem !important; }
h2 { font-size: 1rem !important; font-weight: 500 !important; color: #bbb !important; }
h3 { font-size: 0.72rem !important; font-weight: 500 !important; color: #555 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }

[data-testid="stMetric"] { background: #141414 !important; border: 1px solid #1e1e1e !important; border-radius: 8px !important; padding: 1rem 1.2rem !important; }
[data-testid="stMetricLabel"] p { font-size: 0.7rem !important; color: #555 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
[data-testid="stMetricValue"]   { font-size: 1.5rem !important; font-weight: 600 !important; color: #fff !important; font-family: 'JetBrains Mono', monospace !important; }

.stButton > button { background: #fff !important; color: #000 !important; border: none !important; border-radius: 5px !important; font-size: 0.82rem !important; font-weight: 500 !important; padding: 0.55rem 1.3rem !important; transition: opacity 0.15s !important; }
.stButton > button:hover { opacity: 0.8 !important; }
[data-testid="stDownloadButton"] > button { background: transparent !important; color: #aaa !important; border: 1px solid #2a2a2a !important; border-radius: 5px !important; font-size: 0.8rem !important; width: 100%; }
[data-testid="stDownloadButton"] > button:hover { border-color: #555 !important; }

.stSelectbox > div > div, .stNumberInput input, .stTextInput input, .stDateInput input { background: #141414 !important; border: 1px solid #1e1e1e !important; border-radius: 6px !important; color: #e0e0e0 !important; font-size: 0.83rem !important; }
label { color: #777 !important; font-size: 0.78rem !important; }

[data-testid="stFileUploader"] { background: #141414; border: 1.5px dashed #222; border-radius: 8px; }
[data-testid="stDataFrame"] { border: 1px solid #1e1e1e; border-radius: 8px; }
hr { border-color: #1a1a1a !important; margin: 1.2rem 0 !important; }

.card { background: #141414; border: 1px solid #1e1e1e; border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 0.6rem; }
.card-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.09em; color: #444; margin-bottom: 0.5rem; }
.card-value { font-size: 1.4rem; font-weight: 600; color: #fff; font-family: 'JetBrains Mono', monospace; }
.card-desc  { font-size: 0.8rem; color: #666; margin-top: 0.4rem; line-height: 1.5; }

.sec { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: #3a3a3a; border-bottom: 1px solid #1a1a1a; padding-bottom: 0.4rem; margin: 1.6rem 0 1rem 0; }

.pill { display: inline-block; padding: 0.18rem 0.65rem; border-radius: 99px; font-size: 0.72rem; font-weight: 500; letter-spacing: 0.03em; }
.pl-g { background:#0d2b1a; color:#4ade80; border:1px solid #166534; }
.pl-y { background:#2b2200; color:#facc15; border:1px solid #854d0e; }
.pl-r { background:#2b0d0d; color:#f87171; border:1px solid #991b1b; }

.res-card { background:#141414; border:1px solid #1e1e1e; border-radius:10px; padding:1.6rem 2rem; margin:1rem 0; }
.res-card.churn    { border-left:3px solid #f87171; }
.res-card.no-churn { border-left:3px solid #4ade80; }
</style>
""", unsafe_allow_html=True)

# ── Load bundle ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load("deployment_bundle.pkl")

try:
    bundle = load_bundle()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':'#141414','axes.facecolor':'#141414',
    'axes.edgecolor':'#2a2a2a','axes.labelcolor':'#888',
    'xtick.color':'#555','ytick.color':'#555',
    'text.color':'#ccc','grid.color':'#1e1e1e','grid.linewidth':0.6,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def encode_and_scale(df_input: pd.DataFrame, bundle: dict) -> np.ndarray:
    """
    Encode categorical columns with OrdinalEncoder from bundle['encoders'],
    then scale using the correct per-feature mean_ & scale_ from bundle['scaler'].

    NOTE: bundle['scaler'] was fitted on 25 columns, but the model only uses 15
    (bundle['features']). We therefore extract the relevant mean_ / scale_ by
    matching column names, avoiding a shape mismatch when calling scaler.transform().
    Returns a plain numpy array (model has no feature_names_in_).
    """
    df = df_input.copy()
    encoders      = bundle["encoders"]   # {col: OrdinalEncoder}
    scaler        = bundle["scaler"]     # StandardScaler fitted on 25 cols
    feature_order = bundle["features"]  # 15 top features

    # --- encode categoricals ---
    for col, enc in encoders.items():
        if col in df.columns:
            known = list(enc.categories_[0])
            df[col] = df[col].astype(str).apply(
                lambda x, k=known: x if x in k else k[0]
            )
            df[col] = enc.transform(df[[col]]).ravel()

    # --- reorder to match training feature order ---
    df = df[feature_order]

    # --- manual scale: extract mean_ & scale_ for our 15 features only ---
    scaler_cols = list(scaler.feature_names_in_)
    idx   = [scaler_cols.index(f) for f in feature_order]
    mean_ = scaler.mean_[idx]
    std_  = scaler.scale_[idx]
    arr   = df.values.astype(float)
    return (arr - mean_) / std_


def predict(df_proc: pd.DataFrame, bundle: dict, thr: float):
    model = bundle["model"]
    proba = model.predict_proba(df_proc)[:, 1]
    pred  = (proba >= thr).astype(int)
    risk  = pd.cut(proba, bins=[0, .3, .6, 1.], labels=["Low", "Medium", "High"])
    return proba, pred, risk


def get_cat_options(bundle: dict, col: str) -> list:
    """Return list of category strings for a categorical column."""
    return list(bundle["encoders"][col].categories_[0])


def get_num_range(bundle: dict, col: str) -> dict:
    """Return {min, max, default, is_integer} for a numeric column."""
    return bundle["feature_ranges"][col]

# ── Top bar ───────────────────────────────────────────────────────────────────
col_logo, col_thr = st.columns([3, 1])
with col_logo:
    st.markdown("""
    <div class='topbar'>
        <div>
            <div class='topbar-logo'>◈ Churn Predictor</div>
            <div class='topbar-sub'>Deployment Bundle · Random Forest (Tuned)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_thr:
    st.markdown("<div style='padding-top:1rem'></div>", unsafe_allow_html=True)
    THRESHOLD = st.slider("Threshold", 0.10, 0.90, 0.42, 0.01)

if not model_loaded:
    st.error("deployment_bundle.pkl not found. Pastikan file ada di direktori yang sama dengan app.py.")
    st.stop()

# shortcut references
FEATURES      = bundle["features"]          # 15 fitur
ENCODERS      = bundle["encoders"]
FEATURE_RANGES = bundle["feature_ranges"]
META          = bundle["meta"]

# ── Navigation tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Single Prediction", "Batch Upload", "Model Info"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",     META["model_name"])
    c2.metric("Features",  str(META["n_features"]))
    c3.metric("Threshold", f"{THRESHOLD:.2f}")
    c4.metric("Scenario",  META["scenario"])

    st.markdown("<div class='sec'>How to use</div>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown("""<div class='card'>
            <div class='card-label'>Single Prediction</div>
            <div class='card-desc'>Isi data satu pelanggan lewat form. Dapatkan probabilitas churn dan klasifikasi risiko secara instan.</div>
        </div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("""<div class='card'>
            <div class='card-label'>Batch Upload</div>
            <div class='card-desc'>Upload CSV berisi banyak pelanggan. Download hasil prediksi dengan probabilitas dan label risiko.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec'>Risk labels</div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""<div class='card'>
            <span class='pill pl-g'>Low Risk</span>
            <div class='card-desc' style='margin-top:.6rem'>Probability &lt; 0.30 — pelanggan kemungkinan besar tetap bertahan</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown("""<div class='card'>
            <span class='pill pl-y'>Medium Risk</span>
            <div class='card-desc' style='margin-top:.6rem'>Probability 0.30 – 0.60 — pantau dan pertimbangkan retensi</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown("""<div class='card'>
            <span class='pill pl-r'>High Risk</span>
            <div class='card-desc' style='margin-top:.6rem'>Probability &gt; 0.60 — perlu intervensi segera</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec'>Features used by the model</div>", unsafe_allow_html=True)
    feat_df = pd.DataFrame({
        "#": range(1, len(FEATURES) + 1),
        "Feature": FEATURES,
        "Type": ["Categorical" if f in ENCODERS else "Numerical" for f in FEATURES]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    # ----- build form dynamically from bundle -----
    # Categorical options
    country_opts    = get_cat_options(bundle, "country")
    device_opts     = get_cat_options(bundle, "device_type")
    payment_opts    = get_cat_options(bundle, "payment_method")
    city_opts       = get_cat_options(bundle, "city")
    acq_opts        = get_cat_options(bundle, "acquisition_channel")

    # Numeric ranges
    r_sat    = get_num_range(bundle, "satisfaction_score")
    r_spent  = get_num_range(bundle, "total_spent")
    r_supp   = get_num_range(bundle, "support_tickets")
    r_delay  = get_num_range(bundle, "delivery_delay_days")
    r_mktg   = get_num_range(bundle, "marketing_spend_per_user")
    r_nps    = get_num_range(bundle, "nps_score")
    r_ltv    = get_num_range(bundle, "lifetime_value")
    r_click  = get_num_range(bundle, "email_click_rate")
    r_open   = get_num_range(bundle, "email_open_rate")
    r_sess   = get_num_range(bundle, "avg_session_time")

    with st.form("form_pred"):
        st.markdown("<div class='sec'>Profil & Lokasi</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        country     = c1.selectbox("Country",      country_opts, index=country_opts.index("India") if "India" in country_opts else 0)
        city        = c2.selectbox("City",          city_opts,   index=city_opts.index("Delhi") if "Delhi" in city_opts else 0)
        device_type = c3.selectbox("Device Type",  device_opts)

        st.markdown("<div class='sec'>Akuisisi & Pembayaran</div>", unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        acquisition_channel = c4.selectbox("Acquisition Channel", acq_opts)
        payment_method      = c5.selectbox("Payment Method",      payment_opts)

        st.markdown("<div class='sec'>Finansial</div>", unsafe_allow_html=True)
        c6, c7, c8 = st.columns(3)
        total_spent  = c6.number_input("Total Spent ($)",
                                       min_value=float(r_spent["min"]),
                                       max_value=float(r_spent["max"]),
                                       value=float(r_spent["default"]),
                                       step=1.0)
        lifetime_value = c7.number_input("Lifetime Value ($)",
                                         min_value=float(r_ltv["min"]),
                                         max_value=float(r_ltv["max"]),
                                         value=float(r_ltv["default"]),
                                         step=1.0)
        marketing_spend_per_user = c8.number_input("Marketing Spend / User ($)",
                                                   min_value=float(r_mktg["min"]),
                                                   max_value=float(r_mktg["max"]),
                                                   value=float(r_mktg["default"]),
                                                   step=0.5)

        st.markdown("<div class='sec'>Engagement</div>", unsafe_allow_html=True)
        c9, c10 = st.columns(2)
        email_open_rate  = c9.slider("Email Open Rate",  0.0, float(r_open["max"]),  float(r_open["default"]),  0.01)
        email_click_rate = c10.slider("Email Click Rate", 0.0, float(r_click["max"]), float(r_click["default"]), 0.01)
        avg_session_time = st.number_input("Avg Session Time (min)",
                                           min_value=float(r_sess["min"]),
                                           max_value=float(r_sess["max"]),
                                           value=float(r_sess["default"]),
                                           step=0.1)

        st.markdown("<div class='sec'>Dukungan & Kepuasan</div>", unsafe_allow_html=True)
        c11, c12, c13, c14 = st.columns(4)
        satisfaction_score = c11.slider("Satisfaction (1–5)",
                                        float(r_sat["min"]), float(r_sat["max"]),
                                        float(r_sat["default"]), 0.5)
        nps_score          = c12.slider("NPS Score (0–10)",
                                        int(r_nps["min"]), int(r_nps["max"]),
                                        int(r_nps["default"]))
        support_tickets    = c13.number_input("Support Tickets",
                                              min_value=int(r_supp["min"]),
                                              max_value=int(r_supp["max"]),
                                              value=int(r_supp["default"]),
                                              step=1)
        delivery_delay_days = c14.number_input("Delivery Delay (days)",
                                               min_value=int(r_delay["min"]),
                                               max_value=int(r_delay["max"]),
                                               value=int(r_delay["default"]),
                                               step=1)

        submitted = st.form_submit_button("Run prediction", use_container_width=False)

    if submitted:
        inp = pd.DataFrame([{
            "satisfaction_score":       satisfaction_score,
            "total_spent":              total_spent,
            "support_tickets":          support_tickets,
            "country":                  country,
            "device_type":              device_type,
            "payment_method":           payment_method,
            "city":                     city,
            "delivery_delay_days":      delivery_delay_days,
            "acquisition_channel":      acquisition_channel,
            "marketing_spend_per_user": marketing_spend_per_user,
            "nps_score":                nps_score,
            "lifetime_value":           lifetime_value,
            "email_click_rate":         email_click_rate,
            "email_open_rate":          email_open_rate,
            "avg_session_time":         avg_session_time,
        }])

        df_proc = encode_and_scale(inp, bundle)
        proba, pred, risk = predict(df_proc, bundle, THRESHOLD)

        p = proba[0]; is_churn = pred[0] == 1
        risk_str = str(risk[0])
        pill     = f"<span class='pill {'pl-r' if risk_str=='High' else 'pl-y' if risk_str=='Medium' else 'pl-g'}'>{risk_str} Risk</span>"
        verdict  = "Will Churn" if is_churn else "Will Not Churn"
        vc       = "#f87171" if is_churn else "#4ade80"
        cc       = "churn" if is_churn else "no-churn"

        st.markdown(f"""
        <div class='res-card {cc}'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                <div>
                    <div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.09em;color:#444;margin-bottom:.4rem'>Result</div>
                    <div style='font-size:1.6rem;font-weight:600;color:{vc};font-family:JetBrains Mono,monospace'>{verdict}</div>
                    <div style='margin-top:.7rem'>{pill}</div>
                </div>
                <div style='text-align:right'>
                    <div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.09em;color:#444;margin-bottom:.3rem'>Churn probability</div>
                    <div style='font-size:2.6rem;font-weight:600;color:#fff;font-family:JetBrains Mono,monospace;line-height:1'>{p*100:.1f}%</div>
                    <div style='font-size:.72rem;color:#333;margin-top:.2rem'>threshold · {THRESHOLD:.2f}</div>
                </div>
            </div>
            <div style='margin-top:1.2rem;background:#0d0d0d;border-radius:3px;height:3px'>
                <div style='width:{p*100:.1f}%;height:100%;background:{vc};border-radius:3px'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "<div style='color:#555;font-size:0.83rem;margin-bottom:1.2rem'>"
        "Upload CSV · kolom yang diperlukan: "
        + ", ".join(f"<code>{f}</code>" for f in FEATURES)
        + "</div>",
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_up = pd.read_csv(uploaded)
        has_actual = "churn" in df_up.columns
        y_actual   = df_up["churn"].values if has_actual else None

        # check required columns
        missing_cols = [f for f in FEATURES if f not in df_up.columns]
        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan di CSV: {missing_cols}")
        else:
            with st.spinner("Processing..."):
                df_proc = encode_and_scale(df_up[FEATURES].copy(), bundle)
                proba, pred, risk = predict(df_proc, bundle, THRESHOLD)

            df_res = df_up.copy()
            df_res["churn_probability"] = (proba * 100).round(2)
            df_res["churn_prediction"]  = pred
            df_res["risk_label"]        = risk.astype(str)

            n_total = len(pred); n_churn = pred.sum()
            n_high  = (risk == "High").sum()
            n_med   = (risk == "Medium").sum()
            n_low   = (risk == "Low").sum()

            st.markdown("<div class='sec'>Summary</div>", unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total",       f"{n_total:,}")
            m2.metric("Churn",       f"{n_churn:,}")
            m3.metric("High Risk",   f"{n_high:,}")
            m4.metric("Medium Risk", f"{n_med:,}")
            m5.metric("Low Risk",    f"{n_low:,}")

            st.markdown("<div class='sec'>Distribution</div>", unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 3, figsize=(14, 4)); fig.patch.set_facecolor('#141414')

            # Donut
            ax = axes[0]
            ax.pie([n_total - n_churn, n_churn], colors=['#4ade80', '#f87171'],
                   startangle=90, wedgeprops=dict(width=0.42, edgecolor='#141414', linewidth=2))
            ax.text(0,  0.05, f'{n_churn/n_total*100:.1f}%', ha='center', va='center',
                    fontsize=13, fontweight='600', color='#fff', fontfamily='monospace')
            ax.text(0, -0.15, 'churn rate', ha='center', fontsize=7.5, color='#555')
            ax.set_title('Churn Rate', fontsize=9, color='#666', pad=10, fontweight='400')

            # Histogram
            ax2 = axes[1]
            ax2.hist(proba[pred == 0], bins=25, color='#4ade8022', edgecolor='#4ade80', linewidth=0.5, alpha=0.9, label='No churn')
            ax2.hist(proba[pred == 1], bins=25, color='#f8717122', edgecolor='#f87171', linewidth=0.5, alpha=0.9, label='Churn')
            ax2.axvline(THRESHOLD, color='#ffffff44', linestyle='--', linewidth=1)
            ax2.set_xlabel('Probability', fontsize=8); ax2.grid(True, alpha=0.2)
            ax2.set_title('Probability Distribution', fontsize=9, color='#666', pad=10, fontweight='400')
            ax2.legend(fontsize=7, framealpha=0)

            # Risk bars
            ax3 = axes[2]
            bars = ax3.bar(['Low', 'Medium', 'High'], [n_low, n_med, n_high],
                           color=['#4ade80', '#facc15', '#f87171'], width=0.45,
                           edgecolor='#141414', linewidth=1)
            for bar, val in zip(bars, [n_low, n_med, n_high]):
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(n_low, n_med, n_high) * 0.02,
                         str(val), ha='center', fontsize=8, color='#aaa')
            ax3.set_title('Risk Breakdown', fontsize=9, color='#666', pad=10, fontweight='400')
            ax3.grid(True, alpha=0.2, axis='y')

            plt.tight_layout(pad=2); st.pyplot(fig); plt.close()

            if has_actual:
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                st.markdown("<div class='sec'>Evaluation (vs actual churn column)</div>", unsafe_allow_html=True)
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Accuracy",  f"{accuracy_score(y_actual, pred):.4f}")
                e2.metric("F1-Score",  f"{f1_score(y_actual, pred):.4f}")
                e3.metric("Precision", f"{precision_score(y_actual, pred):.4f}")
                e4.metric("Recall",    f"{recall_score(y_actual, pred):.4f}")

            st.markdown("<div class='sec'>Results</div>", unsafe_allow_html=True)
            show = [c for c in
                    ["country", "city", "device_type", "payment_method",
                     "total_spent", "satisfaction_score", "nps_score",
                     "churn_probability", "churn_prediction", "risk_label"]
                    if c in df_res.columns]
            st.dataframe(df_res[show], use_container_width=True, height=280)
            st.download_button(
                "Download results (.csv)",
                df_res.to_csv(index=False).encode("utf-8"),
                "churn_predictions.csv", "text/csv"
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    t1, t2 = st.tabs(["Feature Importance", "Parameters"])

    with t1:
        model = bundle["model"]
        imp = pd.Series(
            model.feature_importances_,
            index=FEATURES
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(9, 6)); fig.patch.set_facecolor('#141414')
        norm   = plt.Normalize(imp.min(), imp.max())
        colors = plt.cm.Blues(norm(imp.values[::-1]) * 0.6 + 0.3)
        ax.barh(imp.index[::-1], imp.values[::-1], color=colors, height=0.62,
                edgecolor='#141414', linewidth=0.5)
        for i, v in enumerate(imp.values[::-1]):
            ax.text(v + imp.max() * 0.01, i, f'{v:.4f}', va='center', fontsize=8, color='#555')
        ax.set_xlabel('Importance Score', fontsize=8); ax.grid(True, alpha=0.15, axis='x')
        ax.set_title('Feature Importance — All Features', fontsize=10, color='#888', pad=12, fontweight='400')
        ax.set_xlim(0, imp.max() * 1.2)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t2:
        params = bundle["model"].get_params()
        st.dataframe(
            pd.DataFrame({"Parameter": list(params.keys()),
                          "Value": [str(v) for v in params.values()]}),
            use_container_width=True, hide_index=True
        )

        st.markdown("<div class='sec'>All features</div>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame({
                "#": range(1, len(FEATURES) + 1),
                "Feature": FEATURES,
                "Type": ["Categorical" if f in ENCODERS else "Numerical" for f in FEATURES]
            }),
            use_container_width=True, hide_index=True, height=350
        )

        st.markdown("<div class='sec'>Encoder categories</div>", unsafe_allow_html=True)
        for col, enc in ENCODERS.items():
            cats = list(enc.categories_[0])
            st.markdown(f"**{col}**: {', '.join(cats)}")

        st.markdown("<div class='sec'>Numerical feature ranges</div>", unsafe_allow_html=True)
        fr_rows = []
        for f, r in FEATURE_RANGES.items():
            fr_rows.append({"Feature": f, "Min": r["min"], "Max": r["max"],
                            "Default": r["default"], "Is Integer": r["is_integer"]})
        st.dataframe(pd.DataFrame(fr_rows), use_container_width=True, hide_index=True)
