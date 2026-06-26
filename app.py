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

# ── Load bundle ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load("deployment_bundle.pkl")

try:
    bundle       = load_bundle()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Matplotlib theme ─────────────────────────────────────────────────────────
BG   = "#0d0818"
CARD = "#120920"
plt.rcParams.update({
    'figure.facecolor': BG,   'axes.facecolor': CARD,
    'axes.edgecolor':  '#2d1b5e', 'axes.labelcolor': '#9d7fd4',
    'xtick.color': '#6b4fa0',     'ytick.color': '#6b4fa0',
    'text.color':  '#c4b5fd',     'grid.color':  '#1e1240',
    'grid.linewidth': 0.6,        'font.family': 'sans-serif',
})

# ── Helpers ──────────────────────────────────────────────────────────────────
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


def cat_opts(col):
    return list(bundle["encoders"][col].categories_[0])

def num_rng(col):
    return bundle["feature_ranges"][col]


# ── Model check ──────────────────────────────────────────────────────────────
if not model_loaded:
    st.error("⚠️  deployment_bundle.pkl tidak ditemukan. Letakkan file di direktori yang sama dengan app.py.")
    st.stop()

META     = bundle["meta"]
FEATURES = bundle["features"]
ENCODERS = bundle["encoders"]
FR       = bundle["feature_ranges"]

# ── Header ───────────────────────────────────────────────────────────────────
col_hdr, col_thr = st.columns([4, 1])
with col_hdr:
    st.title("🔮 Churn Prediction Web")
    st.caption("Machine Learning · Customer Retention Intelligence")

with col_thr:
    st.write("")
    THRESHOLD = st.slider("⚙ Threshold", 0.10, 0.90, 0.42, 0.01,
                          help="Batas probabilitas untuk klasifikasi churn")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "🔍 Prediksi Single", "📂 Prediksi Batch", "🧠 Info Model"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🤖 Model",     META["model_name"])
    c2.metric("📊 Fitur",     str(META["n_features"]))
    c3.metric("⚡ Skenario",  META["scenario"])
    c4.metric("🎯 Threshold", f"{THRESHOLD:.2f}")

    st.divider()
    st.subheader("Cara Penggunaan")

    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            st.markdown("**🔍 Prediksi Single**")
            st.write("Masukkan data satu pelanggan melalui form interaktif. Sistem akan menghitung probabilitas churn secara real-time beserta klasifikasi risiko Low / Medium / High.")

    with g2:
        with st.container(border=True):
            st.markdown("**📂 Prediksi Batch**")
            st.write("Upload file CSV berisi banyak pelanggan sekaligus. Unduh hasilnya lengkap dengan probabilitas, prediksi, dan label risiko untuk setiap baris data.")

    st.divider()
    st.subheader("Klasifikasi Risiko")

    r1, r2, r3 = st.columns(3)
    with r1:
        with st.container(border=True):
            st.success("🟢 Low Risk")
            st.write("Probabilitas < 0.30\n\nPelanggan sangat mungkin tetap bertahan. Tidak perlu intervensi khusus.")
    with r2:
        with st.container(border=True):
            st.warning("🟡 Medium Risk")
            st.write("Probabilitas 0.30 – 0.60\n\nPantau secara aktif dan pertimbangkan program retensi yang tepat.")
    with r3:
        with st.container(border=True):
            st.error("🔴 High Risk")
            st.write("Probabilitas > 0.60\n\nPerlu intervensi segera. Prioritaskan pelanggan ini dalam strategi retensi.")

    st.divider()
    st.subheader("Fitur yang Digunakan Model")
    feat_df = pd.DataFrame({
        "No":        range(1, len(FEATURES) + 1),
        "Nama Fitur": FEATURES,
        "Tipe":      ["🏷 Kategorikal" if f in ENCODERS else "🔢 Numerikal" for f in FEATURES],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SINGLE PREDICTION
# ════════════════════════════════════════════════════════════════════════════
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
        st.subheader("📍 Lokasi & Profil")
        f1, f2, f3 = st.columns(3)
        country     = f1.selectbox("🌍 Negara",    country_opts,
                                   index=country_opts.index("India") if "India" in country_opts else 0)
        city        = f2.selectbox("🏙 Kota",      city_opts,
                                   index=city_opts.index("Delhi") if "Delhi" in city_opts else 0)
        device_type = f3.selectbox("📱 Perangkat", device_opts)

        st.subheader("📣 Akuisisi & Pembayaran")
        f4, f5 = st.columns(2)
        acquisition_channel = f4.selectbox("📣 Kanal Akuisisi",    acq_opts)
        payment_method      = f5.selectbox("💳 Metode Pembayaran", pay_opts)

        st.subheader("💵 Data Finansial")
        f6, f7, f8 = st.columns(3)
        total_spent    = f6.number_input("💵 Total Belanja ($)",
                                         float(r_spent["min"]), float(r_spent["max"]),
                                         float(r_spent["default"]), step=1.0)
        lifetime_value = f7.number_input("📈 Lifetime Value ($)",
                                         float(r_ltv["min"]), float(r_ltv["max"]),
                                         float(r_ltv["default"]), step=1.0)
        marketing_spend = f8.number_input("📢 Marketing Spend / User ($)",
                                           float(r_mktg["min"]), float(r_mktg["max"]),
                                           float(r_mktg["default"]), step=0.5)

        st.subheader("📧 Engagement")
        f9, f10 = st.columns(2)
        email_open_rate  = f9.slider("📧 Email Open Rate",
                                     0.0, float(r_open["max"]),  float(r_open["default"]),  0.01)
        email_click_rate = f10.slider("🖱 Email Click Rate",
                                      0.0, float(r_click["max"]), float(r_click["default"]), 0.01)
        avg_session_time = st.number_input("⏱ Rata-rata Durasi Sesi (menit)",
                                           float(r_sess["min"]), float(r_sess["max"]),
                                           float(r_sess["default"]), step=0.1)

        st.subheader("⭐ Dukungan & Kepuasan")
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

        submitted = st.form_submit_button("🔮 Jalankan Prediksi", use_container_width=True,
                                          type="primary")

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

        arr                  = encode_and_scale(inp, bundle)
        proba, pred, risk    = predict(arr, bundle, THRESHOLD)
        p                    = float(proba[0])
        is_churn             = bool(pred[0] == 1)
        risk_str             = str(risk[0])

        st.divider()
        st.subheader("Hasil Prediksi")

        res_col, prob_col = st.columns([3, 2])

        with res_col:
            if is_churn:
                st.error(f"⚠️  Diprediksi **CHURN**")
            else:
                st.success(f"✅  Diprediksi **TIDAK CHURN**")

            risk_map = {"Low": "🟢 Low Risk", "Medium": "🟡 Medium Risk", "High": "🔴 High Risk"}
            st.info(f"**Level Risiko:** {risk_map.get(risk_str, risk_str)}")

        with prob_col:
            st.metric("Probabilitas Churn", f"{p*100:.1f}%",
                      help=f"Threshold saat ini: {THRESHOLD:.2f}")

        st.progress(p, text=f"Probabilitas: {p*100:.1f}%  |  Threshold: {THRESHOLD:.2f}")

        # ── Detail card
        with st.expander("📋 Detail Input", expanded=False):
            st.dataframe(inp.T.rename(columns={0: "Nilai"}), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH UPLOAD
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    cols_needed = ", ".join(f"`{f}`" for f in FEATURES)
    st.info(f"**Kolom yang dibutuhkan:** {cols_needed}")

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded:
        df_up   = pd.read_csv(uploaded)
        missing = [f for f in FEATURES if f not in df_up.columns]

        if missing:
            st.error(f"❌ Kolom berikut tidak ditemukan di CSV: {missing}")
        else:
            has_actual = "churn" in df_up.columns
            y_actual   = df_up["churn"].values if has_actual else None

            with st.spinner("Memproses data..."):
                arr               = encode_and_scale(df_up[FEATURES].copy(), bundle)
                proba, pred, risk = predict(arr, bundle, THRESHOLD)

            df_res                      = df_up.copy()
            df_res["churn_probability"] = (proba * 100).round(2)
            df_res["churn_prediction"]  = pred
            df_res["risk_label"]        = risk.astype(str)

            n_total = len(pred)
            n_churn = int(pred.sum())
            n_high  = int((risk == "High").sum())
            n_med   = int((risk == "Medium").sum())
            n_low   = int((risk == "Low").sum())

            # Summary
            st.divider()
            st.subheader("📊 Ringkasan")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Pelanggan",   f"{n_total:,}")
            m2.metric("🔴 Prediksi Churn", f"{n_churn:,}")
            m3.metric("⚠ High Risk",       f"{n_high:,}")
            m4.metric("🟡 Medium Risk",    f"{n_med:,}")
            m5.metric("🟢 Low Risk",       f"{n_low:,}")

            # Charts
            st.divider()
            st.subheader("📈 Distribusi Prediksi")
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            fig.patch.set_facecolor(BG)

            # Donut
            ax = axes[0]
            wedges, _ = ax.pie(
                [n_total - n_churn, n_churn],
                colors=["#4ade80", "#f87171"],
                startangle=90,
                wedgeprops=dict(width=0.45, edgecolor=BG, linewidth=2.5)
            )
            ax.text(0,  .08, f"{n_churn/n_total*100:.1f}%",
                    ha="center", va="center", fontsize=15, fontweight="700",
                    color="#f5f0ff", fontfamily="monospace")
            ax.text(0, -.14, "churn rate", ha="center", fontsize=8, color="#6b4fa0")
            ax.legend(handles=[
                mpatches.Patch(color="#4ade80", label=f"Tidak Churn ({n_total-n_churn})"),
                mpatches.Patch(color="#f87171", label=f"Churn ({n_churn})"),
            ], fontsize=7.5, framealpha=0, loc="lower center", bbox_to_anchor=(.5, -.12))
            ax.set_title("Churn Rate", fontsize=10, color="#9d7fd4", pad=12, fontweight="500")

            # Histogram
            ax2 = axes[1]
            ax2.hist(proba[pred == 0], bins=25,
                     color="#4ade8033", edgecolor="#4ade80", linewidth=.6, alpha=.85, label="Tidak Churn")
            ax2.hist(proba[pred == 1], bins=25,
                     color="#f8717133", edgecolor="#f87171", linewidth=.6, alpha=.85, label="Churn")
            ax2.axvline(THRESHOLD, color="#c4b5fd88", linestyle="--", linewidth=1.2,
                        label=f"Threshold {THRESHOLD:.2f}")
            ax2.set_xlabel("Probabilitas Churn", fontsize=8.5)
            ax2.set_ylabel("Jumlah Pelanggan",  fontsize=8.5)
            ax2.grid(True, alpha=0.18)
            ax2.legend(fontsize=7.5, framealpha=0)
            ax2.set_title("Distribusi Probabilitas", fontsize=10, color="#9d7fd4", pad=12, fontweight="500")

            # Bar
            ax3 = axes[2]
            bars = ax3.bar(["Low", "Medium", "High"], [n_low, n_med, n_high],
                           color=["#4ade80", "#fbbf24", "#f87171"],
                           width=0.42, edgecolor=BG, linewidth=1.5)
            max_val = max(n_low, n_med, n_high) or 1
            for bar, val in zip(bars, [n_low, n_med, n_high]):
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max_val * 0.03,
                         str(val), ha="center", fontsize=9.5,
                         color="#c4b5fd", fontweight="600")
            ax3.set_ylabel("Jumlah Pelanggan", fontsize=8.5)
            ax3.grid(True, alpha=0.18, axis="y")
            ax3.set_title("Distribusi Level Risiko", fontsize=10,
                          color="#9d7fd4", pad=12, fontweight="500")
            ax3.set_ylim(0, max_val * 1.2)

            plt.tight_layout(pad=2.5)
            st.pyplot(fig)
            plt.close()

            # Evaluasi
            if has_actual:
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                st.divider()
                st.subheader("✅ Evaluasi vs Data Aktual")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Accuracy",  f"{accuracy_score(y_actual, pred):.4f}")
                e2.metric("F1-Score",  f"{f1_score(y_actual, pred):.4f}")
                e3.metric("Precision", f"{precision_score(y_actual, pred):.4f}")
                e4.metric("Recall",    f"{recall_score(y_actual, pred):.4f}")

            # Tabel
            st.divider()
            st.subheader("📋 Tabel Hasil Prediksi")
            show = [c for c in
                    ["country", "city", "device_type", "payment_method",
                     "total_spent", "satisfaction_score", "nps_score",
                     "churn_probability", "churn_prediction", "risk_label"]
                    if c in df_res.columns]
            st.dataframe(df_res[show], use_container_width=True, height=300)

            dl_col, _ = st.columns([1, 3])
            with dl_col:
                st.download_button(
                    "⬇ Unduh Hasil (.csv)",
                    df_res.to_csv(index=False).encode("utf-8"),
                    "churn_predictions.csv", "text/csv"
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    sub1, sub2 = st.tabs(["📊 Feature Importance", "⚙ Parameter & Detail"])

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
                       color=colors, height=0.6, edgecolor=BG, linewidth=.8)
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
        st.subheader("⚙ Parameter Model")
        params = bundle["model"].get_params()
        st.dataframe(
            pd.DataFrame({"Parameter": list(params.keys()),
                          "Nilai":     [str(v) for v in params.values()]}),
            use_container_width=True, hide_index=True
        )

        st.divider()
        st.subheader("📋 Daftar Fitur")
        st.dataframe(
            pd.DataFrame({
                "No":     range(1, len(FEATURES) + 1),
                "Fitur":  FEATURES,
                "Tipe":   ["🏷 Kategorikal" if f in ENCODERS else "🔢 Numerikal" for f in FEATURES],
            }),
            use_container_width=True, hide_index=True, height=360
        )

        st.divider()
        st.subheader("🏷 Kategori Encoder")
        rows = []
        for col, enc in ENCODERS.items():
            cats = list(enc.categories_[0])
            rows.append({"Kolom": col, "Kategori": ", ".join(cats)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🔢 Rentang Nilai Fitur Numerikal")
        fr_rows = [{"Fitur": f, "Min": round(r["min"], 3), "Max": round(r["max"], 3),
                    "Default": round(r["default"], 3), "Integer?": r["is_integer"]}
                   for f, r in FR.items()]
        st.dataframe(pd.DataFrame(fr_rows), use_container_width=True, hide_index=True)
