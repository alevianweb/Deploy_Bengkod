import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Churn Prediction Web",
    page_icon="🔮",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES  —  Deep Purple Glassmorphism
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2.5rem 2rem 2.5rem !important;
    max-width: 1280px !important;
}

/* ── Background: deep purple mesh ── */
.stApp {
    background: #0a0612 !important;
    background-image:
        radial-gradient(ellipse 80% 60% at 10% 0%,  #2d1b6944 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 10%, #5b21b622 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 90%, #1e0b4422 0%, transparent 50%) !important;
    color: #e2d9f3 !important;
}

/* ── Header banner ── */
.header-wrap {
    background: linear-gradient(135deg, rgba(109,40,217,.18) 0%, rgba(76,29,149,.10) 100%);
    border: 1px solid rgba(139,92,246,.25);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin: 1.4rem 0 1.8rem 0;
    backdrop-filter: blur(12px);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.brand-row { display: flex; align-items: center; gap: .9rem; }
.brand-icon {
    width: 42px; height: 42px; border-radius: 10px;
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; box-shadow: 0 0 18px #7c3aed66;
}
.brand-name {
    font-size: 1.15rem; font-weight: 700; color: #f5f0ff;
    letter-spacing: -.02em; line-height: 1.1;
}
.brand-sub  { font-size: .7rem; color: #7c5bba; margin-top: .15rem; }
.badge {
    font-size: .68rem; font-weight: 600; letter-spacing: .07em;
    text-transform: uppercase; padding: .3rem .8rem;
    border-radius: 99px; background: rgba(124,58,237,.2);
    border: 1px solid rgba(139,92,246,.4); color: #c4b5fd;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(109,40,217,.08) !important;
    border-radius: 10px !important;
    padding: .3rem !important;
    gap: .2rem !important;
    border: 1px solid rgba(139,92,246,.15) !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #7c5bba !important;
    font-size: .78rem !important; font-weight: 500 !important;
    letter-spacing: .04em !important; text-transform: uppercase !important;
    padding: .55rem 1.2rem !important;
    border-radius: 7px !important; border: none !important;
    transition: all .2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(124,58,237,.55), rgba(109,40,217,.4)) !important;
    color: #e9d5ff !important; font-weight: 600 !important;
    box-shadow: 0 2px 12px rgba(124,58,237,.3) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 0 !important; }

/* ── Glass metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(139,92,246,.2) !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.3rem !important;
    backdrop-filter: blur(10px) !important;
    transition: border-color .2s !important;
}
[data-testid="stMetric"]:hover { border-color: rgba(139,92,246,.45) !important; }
[data-testid="stMetricLabel"] p {
    font-size: .68rem !important; color: #7c5bba !important;
    text-transform: uppercase !important; letter-spacing: .07em !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.45rem !important; font-weight: 700 !important;
    color: #e9d5ff !important; font-family: 'Space Mono', monospace !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #9333ea) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-size: .82rem !important;
    font-weight: 600 !important; padding: .6rem 1.6rem !important;
    letter-spacing: .02em !important;
    box-shadow: 0 4px 15px rgba(124,58,237,.35) !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,.5) !important;
}
[data-testid="stDownloadButton"] > button {
    background: rgba(124,58,237,.12) !important; color: #c4b5fd !important;
    border: 1px solid rgba(139,92,246,.3) !important;
    border-radius: 8px !important; font-size: .8rem !important; width: 100%;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(124,58,237,.22) !important;
    border-color: rgba(139,92,246,.6) !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput input,
.stNumberInput > div > div > input,
.stTextInput input,
input[type="number"],
input[type="text"] {
    background: rgba(30, 15, 60, 0.85) !important;
    border: 1px solid rgba(139,92,246,.35) !important;
    border-radius: 8px !important;
    color: #e2d9f3 !important;
    font-size: .83rem !important;
    caret-color: #c4b5fd !important;
}
/* placeholder text */
input::placeholder { color: #7c5bba !important; opacity: 1 !important; }
input[type="number"]::placeholder { color: #7c5bba !important; opacity: 1 !important; }

/* selectbox displayed value text */
.stSelectbox > div > div > div,
.stSelectbox span,
[data-baseweb="select"] span,
[data-baseweb="select"] div { color: #e2d9f3 !important; }

/* number input container */
[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] > div > div > input {
    background: rgba(30, 15, 60, 0.85) !important;
    color: #e2d9f3 !important;
}
/* +/- buttons on number input */
[data-testid="stNumberInput"] button {
    background: rgba(124,58,237,.2) !important;
    color: #c4b5fd !important;
    border: 1px solid rgba(139,92,246,.3) !important;
}
[data-testid="stNumberInput"] button:hover {
    background: rgba(124,58,237,.4) !important;
}

.stSelectbox > div > div:focus-within,
.stNumberInput input:focus {
    border-color: rgba(139,92,246,.7) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,.2) !important;
}
label, .stSelectbox label { color: #9d7fd4 !important; font-size: .77rem !important; }

/* ── Slider ── */
.stSlider [data-testid="stTickBar"] { color: #6b4fa0 !important; }
/* nilai angka di atas thumb slider */
.stSlider [data-testid="stThumbValue"],
.stSlider div[data-testid="stThumbValue"] > div,
.stSlider output,
.stSlider p { color: #e2d9f3 !important; font-size: .82rem !important; }
/* track warna ungu */
.stSlider [role="slider"] { background: #7c3aed !important; border-color: #a855f7 !important; }

/* ── Form submit button ── */
[data-testid="stForm"] button[kind="primaryFormSubmit"],
[data-testid="stForm"] .stButton > button,
button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, #7c3aed, #9333ea) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
}

/* ── File uploader — hilangkan teks duplikat ── */
[data-testid="stFileUploader"] {
    background: rgba(124,58,237,.06) !important;
    border: 1.5px dashed rgba(139,92,246,.35) !important;
    border-radius: 12px !important;
}
/* sembunyikan label tersembunyi yang menyebabkan double text */
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span { color: #9d7fd4 !important; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }
/* tombol Browse di uploader */
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(124,58,237,.25) !important;
    color: #e2d9f3 !important;
    border: 1px solid rgba(139,92,246,.4) !important;
    border-radius: 6px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(139,92,246,.2) !important;
    border-radius: 10px !important;
}

/* ── Section divider ── */
.sec-label {
    font-size: .65rem; text-transform: uppercase; letter-spacing: .12em;
    color: #6b4fa0; border-bottom: 1px solid rgba(139,92,246,.18);
    padding-bottom: .35rem; margin: 1.8rem 0 1rem 0;
    display: flex; align-items: center; gap: .5rem;
}
.sec-label::before {
    content: ''; display: inline-block;
    width: 3px; height: 12px; border-radius: 2px;
    background: linear-gradient(#7c3aed, #a855f7);
}

/* ── Glass cards ── */
.gcard {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(139,92,246,.18);
    border-radius: 12px; padding: 1.3rem 1.5rem; margin-bottom: .7rem;
    backdrop-filter: blur(8px);
    transition: border-color .2s, transform .2s;
}
.gcard:hover { border-color: rgba(139,92,246,.4); transform: translateY(-2px); }
.gcard-icon  { font-size: 1.4rem; margin-bottom: .5rem; }
.gcard-title { font-size: .78rem; font-weight: 600; color: #c4b5fd;
               text-transform: uppercase; letter-spacing: .06em; margin-bottom: .4rem; }
.gcard-body  { font-size: .82rem; color: #9d7fd4; line-height: 1.6; }

/* ── Risk pills ── */
.pill {
    display: inline-flex; align-items: center; gap: .3rem;
    padding: .2rem .75rem; border-radius: 99px;
    font-size: .72rem; font-weight: 600; letter-spacing: .04em;
}
.pl-g { background: rgba(34,197,94,.12);  color: #4ade80; border: 1px solid rgba(34,197,94,.3); }
.pl-y { background: rgba(234,179,8,.12);  color: #fbbf24; border: 1px solid rgba(234,179,8,.3); }
.pl-r { background: rgba(239,68,68,.12);  color: #f87171; border: 1px solid rgba(239,68,68,.3); }

/* ── Result card ── */
.result-wrap {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(139,92,246,.22);
    border-radius: 16px; padding: 2rem 2.2rem; margin: 1.2rem 0;
    backdrop-filter: blur(14px);
}
.result-wrap.is-churn    { border-left: 4px solid #f87171; background: rgba(239,68,68,.05); }
.result-wrap.no-churn    { border-left: 4px solid #4ade80; background: rgba(34,197,94,.05); }

/* ── Spinner ── */
.stSpinner > div { border-color: #7c3aed transparent transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3b1d7a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Load bundle ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load("deployment_bundle.pkl")

try:
    bundle       = load_bundle()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Matplotlib purple-glass theme ─────────────────────────────────────────────
BG   = "#0d0818"
CARD = "#120920"
plt.rcParams.update({
    'figure.facecolor': BG,   'axes.facecolor': CARD,
    'axes.edgecolor':  '#2d1b5e', 'axes.labelcolor': '#9d7fd4',
    'xtick.color': '#6b4fa0',     'ytick.color': '#6b4fa0',
    'text.color':  '#c4b5fd',     'grid.color':  '#1e1240',
    'grid.linewidth': 0.6,        'font.family': 'sans-serif',
})

# ── Core helpers ───────────────────────────────────────────────────────────────
def encode_and_scale(df_input: pd.DataFrame, bundle: dict) -> np.ndarray:
    df            = df_input.copy()
    encoders      = bundle["encoders"]
    scaler        = bundle["scaler"]
    feature_order = bundle["features"]

    for col, enc in encoders.items():
        if col in df.columns:
            known   = list(enc.categories_[0])
            df[col] = df[col].astype(str).apply(lambda x, k=known: x if x in k else k[0])
            df[col] = enc.transform(df[[col]]).ravel()

    df          = df[feature_order]
    scaler_cols = list(scaler.feature_names_in_)
    idx         = [scaler_cols.index(f) for f in feature_order]
    mean_       = scaler.mean_[idx]
    std_        = scaler.scale_[idx]
    return (df.values.astype(float) - mean_) / std_


def predict(arr: np.ndarray, bundle: dict, thr: float):
    proba = bundle["model"].predict_proba(arr)[:, 1]
    pred  = (proba >= thr).astype(int)
    risk  = pd.cut(proba, bins=[0, .3, .6, 1.], labels=["Low", "Medium", "High"])
    return proba, pred, risk


def cat_opts(col):  return list(bundle["encoders"][col].categories_[0])
def num_rng(col):   return bundle["feature_ranges"][col]

# ── HEADER ─────────────────────────────────────────────────────────────────────
if model_loaded:
    META      = bundle["meta"]
    FEATURES  = bundle["features"]
    ENCODERS  = bundle["encoders"]
    FR        = bundle["feature_ranges"]

col_hdr, col_thr = st.columns([4, 1])
with col_hdr:
    st.markdown("""
    <div class='header-wrap'>
        <div class='brand-row'>
            <div class='brand-icon'>🔮</div>
            <div>
                <div class='brand-name'>Churn Prediction Web</div>
                <div class='brand-sub'>Machine Learning · Customer Retention Intelligence</div>
            </div>
        </div>
        <span class='badge'>Random Forest · Tuned</span>
    </div>
    """, unsafe_allow_html=True)

with col_thr:
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    THRESHOLD = st.slider("⚙ Threshold", 0.10, 0.90, 0.42, 0.01,
                          help="Batas probabilitas untuk klasifikasi churn")

if not model_loaded:
    st.error("⚠️  deployment_bundle.pkl tidak ditemukan. Letakkan file di direktori yang sama dengan app.py.")
    st.stop()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🏠  Dashboard", "🔍  Prediksi Single", "📂  Prediksi Batch", "🧠  Info Model"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD / OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🤖  Model",    META["model_name"])
    c2.metric("📊  Fitur",    str(META["n_features"]))
    c3.metric("⚡  Skenario", META["scenario"])
    c4.metric("🎯  Threshold", f"{THRESHOLD:.2f}")

    st.markdown("<div class='sec-label'>Cara Penggunaan</div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("""<div class='gcard'>
            <div class='gcard-icon'>🔍</div>
            <div class='gcard-title'>Prediksi Single</div>
            <div class='gcard-body'>Masukkan data satu pelanggan melalui form interaktif. Sistem akan menghitung probabilitas churn secara real-time beserta klasifikasi risiko Low / Medium / High.</div>
        </div>""", unsafe_allow_html=True)
    with g2:
        st.markdown("""<div class='gcard'>
            <div class='gcard-icon'>📂</div>
            <div class='gcard-title'>Prediksi Batch</div>
            <div class='gcard-body'>Upload file CSV berisi banyak pelanggan sekaligus. Unduh hasilnya lengkap dengan probabilitas, prediksi, dan label risiko untuk setiap baris data.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>Klasifikasi Risiko</div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""<div class='gcard'>
            <span class='pill pl-g'>🟢  Low Risk</span>
            <div class='gcard-body' style='margin-top:.7rem'>Probabilitas &lt; 0.30<br>Pelanggan sangat mungkin tetap bertahan. Tidak perlu intervensi khusus.</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown("""<div class='gcard'>
            <span class='pill pl-y'>🟡  Medium Risk</span>
            <div class='gcard-body' style='margin-top:.7rem'>Probabilitas 0.30 – 0.60<br>Pantau secara aktif dan pertimbangkan program retensi yang tepat.</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown("""<div class='gcard'>
            <span class='pill pl-r'>🔴  High Risk</span>
            <div class='gcard-body' style='margin-top:.7rem'>Probabilitas &gt; 0.60<br>Perlu intervensi segera. Prioritaskan pelanggan ini dalam strategi retensi.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>Fitur yang Digunakan Model</div>", unsafe_allow_html=True)
    feat_df = pd.DataFrame({
        "No": range(1, len(FEATURES) + 1),
        "Nama Fitur": FEATURES,
        "Tipe": ["🏷 Kategorikal" if f in ENCODERS else "🔢 Numerikal" for f in FEATURES],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    country_opts = cat_opts("country")
    device_opts  = cat_opts("device_type")
    pay_opts     = cat_opts("payment_method")
    city_opts    = cat_opts("city")
    acq_opts     = cat_opts("acquisition_channel")

    r_sat   = num_rng("satisfaction_score")
    r_spent = num_rng("total_spent")
    r_supp  = num_rng("support_tickets")
    r_delay = num_rng("delivery_delay_days")
    r_mktg  = num_rng("marketing_spend_per_user")
    r_nps   = num_rng("nps_score")
    r_ltv   = num_rng("lifetime_value")
    r_click = num_rng("email_click_rate")
    r_open  = num_rng("email_open_rate")
    r_sess  = num_rng("avg_session_time")

    with st.form("single_form"):
        st.markdown("<div class='sec-label'>Lokasi & Profil</div>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        country     = f1.selectbox("🌍 Negara",       country_opts,
                                   index=country_opts.index("India") if "India" in country_opts else 0)
        city        = f2.selectbox("🏙 Kota",         city_opts,
                                   index=city_opts.index("Delhi") if "Delhi" in city_opts else 0)
        device_type = f3.selectbox("📱 Perangkat",    device_opts)

        st.markdown("<div class='sec-label'>Akuisisi & Pembayaran</div>", unsafe_allow_html=True)
        f4, f5 = st.columns(2)
        acquisition_channel = f4.selectbox("📣 Kanal Akuisisi",  acq_opts)
        payment_method      = f5.selectbox("💳 Metode Pembayaran", pay_opts)

        st.markdown("<div class='sec-label'>Data Finansial</div>", unsafe_allow_html=True)
        f6, f7, f8 = st.columns(3)
        total_spent  = f6.number_input("💵 Total Belanja ($)",
                                       float(r_spent["min"]), float(r_spent["max"]),
                                       float(r_spent["default"]), step=1.0)
        lifetime_value = f7.number_input("📈 Lifetime Value ($)",
                                         float(r_ltv["min"]), float(r_ltv["max"]),
                                         float(r_ltv["default"]), step=1.0)
        marketing_spend = f8.number_input("📢 Marketing Spend / User ($)",
                                          float(r_mktg["min"]), float(r_mktg["max"]),
                                          float(r_mktg["default"]), step=0.5)

        st.markdown("<div class='sec-label'>Keterlibatan (Engagement)</div>", unsafe_allow_html=True)
        f9, f10 = st.columns(2)
        email_open_rate  = f9.slider("📧 Email Open Rate",
                                     0.0, float(r_open["max"]),  float(r_open["default"]),  0.01)
        email_click_rate = f10.slider("🖱 Email Click Rate",
                                      0.0, float(r_click["max"]), float(r_click["default"]), 0.01)
        avg_session_time = st.number_input("⏱ Rata-rata Durasi Sesi (menit)",
                                           float(r_sess["min"]), float(r_sess["max"]),
                                           float(r_sess["default"]), step=0.1)

        st.markdown("<div class='sec-label'>Dukungan & Kepuasan Pelanggan</div>", unsafe_allow_html=True)
        f11, f12, f13, f14 = st.columns(4)
        satisfaction_score  = f11.slider("⭐ Skor Kepuasan (1–5)",
                                         float(r_sat["min"]), float(r_sat["max"]),
                                         float(r_sat["default"]), 0.5)
        nps_score           = f12.slider("📊 NPS Score (0–10)",
                                         int(r_nps["min"]), int(r_nps["max"]), int(r_nps["default"]))
        support_tickets     = f13.number_input("🎫 Tiket Support",
                                               int(r_supp["min"]), int(r_supp["max"]),
                                               int(r_supp["default"]), step=1)
        delivery_delay_days = f14.number_input("🚚 Keterlambatan Pengiriman (hari)",
                                               int(r_delay["min"]), int(r_delay["max"]),
                                               int(r_delay["default"]), step=1)

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮  Jalankan Prediksi", use_container_width=False)

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
            "marketing_spend_per_user": marketing_spend,
            "nps_score":                nps_score,
            "lifetime_value":           lifetime_value,
            "email_click_rate":         email_click_rate,
            "email_open_rate":          email_open_rate,
            "avg_session_time":         avg_session_time,
        }])

        arr  = encode_and_scale(inp, bundle)
        proba, pred, risk = predict(arr, bundle, THRESHOLD)

        p        = float(proba[0])
        is_churn = bool(pred[0] == 1)
        risk_str = str(risk[0])

        # ── warna berdasarkan risk & prediksi ──
        vc         = "#f87171" if is_churn else "#4ade80"
        card_bg    = "rgba(239,68,68,.07)"    if is_churn else "rgba(34,197,94,.07)"
        card_bdr   = "#f87171"                if is_churn else "#4ade80"
        verdict    = "⚠ Diprediksi CHURN"    if is_churn else "✓ Diprediksi TIDAK CHURN"

        if risk_str == "High":
            pill_bg, pill_color, pill_bdr, pill_ico = "rgba(239,68,68,.15)", "#f87171", "rgba(239,68,68,.4)", "🔴"
        elif risk_str == "Medium":
            pill_bg, pill_color, pill_bdr, pill_ico = "rgba(234,179,8,.15)",  "#fbbf24", "rgba(234,179,8,.4)",  "🟡"
        else:
            pill_bg, pill_color, pill_bdr, pill_ico = "rgba(34,197,94,.15)",  "#4ade80", "rgba(34,197,94,.4)",  "🟢"

        st.markdown(f"""
        <div style="
            background: {card_bg};
            border: 1px solid rgba(139,92,246,.25);
            border-left: 4px solid {card_bdr};
            border-radius: 16px;
            padding: 2rem 2.2rem;
            margin: 1.4rem 0;
            backdrop-filter: blur(14px);
        ">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1.2rem;">

                <div>
                    <div style="font-size:.63rem; text-transform:uppercase; letter-spacing:.12em;
                                color:#7c5bba; margin-bottom:.6rem; font-family:'Sora',sans-serif;">
                        Hasil Prediksi
                    </div>
                    <div style="font-size:1.75rem; font-weight:700; color:{vc};
                                font-family:'Space Mono',monospace; line-height:1.15;">
                        {verdict}
                    </div>
                    <div style="margin-top:.9rem;">
                        <span style="
                            display:inline-flex; align-items:center; gap:.35rem;
                            padding:.25rem .85rem; border-radius:99px;
                            font-size:.75rem; font-weight:700; letter-spacing:.05em;
                            font-family:'Sora',sans-serif;
                            background:{pill_bg}; color:{pill_color}; border:1px solid {pill_bdr};
                        ">{pill_ico} {risk_str} Risk</span>
                    </div>
                </div>

                <div style="text-align:right;">
                    <div style="font-size:.63rem; text-transform:uppercase; letter-spacing:.12em;
                                color:#7c5bba; margin-bottom:.5rem; font-family:'Sora',sans-serif;">
                        Probabilitas Churn
                    </div>
                    <div style="font-size:3.2rem; font-weight:700; color:#f5f0ff;
                                font-family:'Space Mono',monospace; line-height:1;">
                        {p*100:.1f}<span style="font-size:1.5rem; color:#c4b5fd;">%</span>
                    </div>
                    <div style="font-size:.7rem; color:#6b4fa0; margin-top:.35rem;
                                font-family:'Sora',sans-serif;">
                        threshold · {THRESHOLD:.2f}
                    </div>
                </div>

            </div>

            <!-- progress bar -->
            <div style="margin-top:1.5rem; background:rgba(255,255,255,.07);
                        border-radius:6px; height:6px; overflow:hidden;">
                <div style="width:{p*100:.1f}%; height:100%; border-radius:6px;
                            background:linear-gradient(90deg, {vc}88, {vc});"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:.4rem;
                        font-size:.65rem; color:#4b3680; font-family:'Sora',sans-serif;">
                <span>0%</span><span>50%</span><span>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    cols_needed = " · ".join(f"`{f}`" for f in FEATURES)
    st.markdown(
        f"<div style='color:#7c5bba;font-size:.8rem;margin-bottom:1.2rem;line-height:1.7'>"
        f"Upload file CSV · Kolom yang dibutuhkan:<br>{cols_needed}</div>",
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_up  = pd.read_csv(uploaded)
        missing = [f for f in FEATURES if f not in df_up.columns]

        if missing:
            st.error(f"❌  Kolom berikut tidak ditemukan di CSV: {missing}")
        else:
            has_actual = "churn" in df_up.columns
            y_actual   = df_up["churn"].values if has_actual else None

            with st.spinner("Memproses data..."):
                arr  = encode_and_scale(df_up[FEATURES].copy(), bundle)
                proba, pred, risk = predict(arr, bundle, THRESHOLD)

            df_res = df_up.copy()
            df_res["churn_probability"] = (proba * 100).round(2)
            df_res["churn_prediction"]  = pred
            df_res["risk_label"]        = risk.astype(str)

            n_total = len(pred); n_churn = int(pred.sum())
            n_high  = int((risk == "High").sum())
            n_med   = int((risk == "Medium").sum())
            n_low   = int((risk == "Low").sum())

            # ── Summary metrics
            st.markdown("<div class='sec-label'>Ringkasan</div>", unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Pelanggan", f"{n_total:,}")
            m2.metric("🔴 Prediksi Churn", f"{n_churn:,}")
            m3.metric("⚠ High Risk",   f"{n_high:,}")
            m4.metric("🟡 Medium Risk",  f"{n_med:,}")
            m5.metric("🟢 Low Risk",     f"{n_low:,}")

            # ── Charts
            st.markdown("<div class='sec-label'>Distribusi Prediksi</div>", unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            fig.patch.set_facecolor(BG)

            # Donut churn rate
            ax = axes[0]
            wedge_colors = ["#4ade80", "#f87171"]
            wedges, _ = ax.pie(
                [n_total - n_churn, n_churn], colors=wedge_colors,
                startangle=90,
                wedgeprops=dict(width=0.45, edgecolor=BG, linewidth=2.5)
            )
            ax.text(0,  .08, f"{n_churn/n_total*100:.1f}%",
                    ha="center", va="center", fontsize=15, fontweight="700",
                    color="#f5f0ff", fontfamily="monospace")
            ax.text(0, -.14, "churn rate",
                    ha="center", fontsize=8, color="#6b4fa0")
            legend_handles = [
                mpatches.Patch(color="#4ade80", label=f"Tidak Churn ({n_total-n_churn})"),
                mpatches.Patch(color="#f87171", label=f"Churn ({n_churn})"),
            ]
            ax.legend(handles=legend_handles, fontsize=7.5, framealpha=0,
                      loc="lower center", bbox_to_anchor=(.5, -.12))
            ax.set_title("Churn Rate", fontsize=10, color="#9d7fd4", pad=12, fontweight="500")

            # Histogram distribusi probabilitas
            ax2 = axes[1]
            ax2.hist(proba[pred == 0], bins=25,
                     color="#4ade8033", edgecolor="#4ade80", linewidth=.6, alpha=.85, label="Tidak Churn")
            ax2.hist(proba[pred == 1], bins=25,
                     color="#f8717133", edgecolor="#f87171", linewidth=.6, alpha=.85, label="Churn")
            ax2.axvline(THRESHOLD, color="#c4b5fd88", linestyle="--", linewidth=1.2, label=f"Threshold {THRESHOLD:.2f}")
            ax2.set_xlabel("Probabilitas Churn", fontsize=8.5)
            ax2.set_ylabel("Jumlah Pelanggan", fontsize=8.5)
            ax2.grid(True, alpha=0.18)
            ax2.legend(fontsize=7.5, framealpha=0)
            ax2.set_title("Distribusi Probabilitas", fontsize=10, color="#9d7fd4", pad=12, fontweight="500")

            # Bar chart risk level
            ax3 = axes[2]
            bar_colors = ["#4ade80", "#fbbf24", "#f87171"]
            bars = ax3.bar(["Low", "Medium", "High"], [n_low, n_med, n_high],
                           color=bar_colors, width=0.42,
                           edgecolor=BG, linewidth=1.5)
            max_val = max(n_low, n_med, n_high) or 1
            for bar, val in zip(bars, [n_low, n_med, n_high]):
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max_val * 0.03,
                         str(val), ha="center", fontsize=9.5,
                         color="#c4b5fd", fontweight="600")
            ax3.set_ylabel("Jumlah Pelanggan", fontsize=8.5)
            ax3.grid(True, alpha=0.18, axis="y")
            ax3.set_title("Distribusi Level Risiko", fontsize=10, color="#9d7fd4", pad=12, fontweight="500")
            ax3.set_ylim(0, max_val * 1.2)

            plt.tight_layout(pad=2.5)
            st.pyplot(fig)
            plt.close()

            # ── Evaluation (jika ada kolom churn aktual)
            if has_actual:
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                st.markdown("<div class='sec-label'>Evaluasi vs Data Aktual</div>", unsafe_allow_html=True)
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("✅ Accuracy",  f"{accuracy_score(y_actual, pred):.4f}")
                e2.metric("⚖ F1-Score",   f"{f1_score(y_actual, pred):.4f}")
                e3.metric("🎯 Precision", f"{precision_score(y_actual, pred):.4f}")
                e4.metric("📡 Recall",    f"{recall_score(y_actual, pred):.4f}")

            # ── Results table
            st.markdown("<div class='sec-label'>Tabel Hasil Prediksi</div>", unsafe_allow_html=True)
            show = [c for c in
                    ["country", "city", "device_type", "payment_method",
                     "total_spent", "satisfaction_score", "nps_score",
                     "churn_probability", "churn_prediction", "risk_label"]
                    if c in df_res.columns]
            st.dataframe(df_res[show], use_container_width=True, height=300)

            dl_col, _ = st.columns([1, 3])
            with dl_col:
                st.download_button(
                    "⬇  Unduh Hasil (.csv)",
                    df_res.to_csv(index=False).encode("utf-8"),
                    "churn_predictions.csv", "text/csv"
                )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    sub1, sub2 = st.tabs(["📊  Feature Importance", "⚙  Parameter & Detail"])

    with sub1:
        imp = pd.Series(
            bundle["model"].feature_importances_, index=FEATURES
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(BG)

        norm   = plt.Normalize(imp.min(), imp.max())
        cmap   = plt.colormaps["RdPu"]
        colors = [cmap(norm(v) * .7 + .25) for v in imp.values[::-1]]

        bars = ax.barh(imp.index[::-1], imp.values[::-1],
                       color=colors, height=0.6,
                       edgecolor=BG, linewidth=.8)
        for i, v in enumerate(imp.values[::-1]):
            ax.text(v + imp.max() * .012, i, f"{v:.4f}",
                    va="center", fontsize=8, color="#c4b5fd")

        ax.set_xlabel("Importance Score", fontsize=9)
        ax.grid(True, alpha=0.15, axis="x")
        ax.set_title("Feature Importance — Semua Fitur", fontsize=11,
                     color="#c4b5fd", pad=14, fontweight="500")
        ax.set_xlim(0, imp.max() * 1.22)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with sub2:
        st.markdown("<div class='sec-label'>Parameter Model</div>", unsafe_allow_html=True)
        params = bundle["model"].get_params()
        st.dataframe(
            pd.DataFrame({"Parameter": list(params.keys()),
                          "Nilai":     [str(v) for v in params.values()]}),
            use_container_width=True, hide_index=True
        )

        st.markdown("<div class='sec-label'>Daftar Fitur</div>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame({
                "No":       range(1, len(FEATURES) + 1),
                "Fitur":    FEATURES,
                "Tipe":     ["🏷 Kategorikal" if f in ENCODERS else "🔢 Numerikal" for f in FEATURES],
            }),
            use_container_width=True, hide_index=True, height=360
        )

        st.markdown("<div class='sec-label'>Kategori Encoder</div>", unsafe_allow_html=True)
        for col, enc in ENCODERS.items():
            cats = list(enc.categories_[0])
            st.markdown(
                f"<span style='color:#c4b5fd;font-weight:600'>{col}</span>"
                f"<span style='color:#6b4fa0'> → </span>"
                f"<span style='color:#9d7fd4'>{', '.join(cats)}</span>",
                unsafe_allow_html=True
            )

        st.markdown("<div class='sec-label'>Rentang Nilai Fitur Numerikal</div>", unsafe_allow_html=True)
        fr_rows = [{"Fitur": f, "Min": round(r["min"], 3), "Max": round(r["max"], 3),
                    "Default": round(r["default"], 3), "Integer?": r["is_integer"]}
                   for f, r in FR.items()]
        st.dataframe(pd.DataFrame(fr_rows), use_container_width=True, hide_index=True)
